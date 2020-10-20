import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb

from pathlib import Path
from joblib import Parallel, delayed
from sklearn.neighbors import kneighbors_graph

sys.path.append('../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *


class ImageIndexer(object):
    def __init__(self, db_path, buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.gradient_arrays_db = None
        self.gradient_shapes_db = None
        self.idxs = {"index": 0}

        self.gradient_arrays_buffer = []
        self.gradient_shapes_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gradient_arrays_buffer:
            print("writing last buffers")
            print(len(self.gradient_arrays_buffer))

            self._write_buffer(self.gradient_arrays_db, self.gradient_arrays_buffer)
            self._write_buffer(self.gradient_shapes_db, self.gradient_shapes_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.gradient_arrays_db = self.db.create_dataset(
            "perceptual_gradients",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('float32'))
        )

        self.gradient_shapes_db = self.db.create_dataset(
            "gradient_shapes",
            shape=(self.num_of_images, 2),
            maxshape=(None, 2),
            dtype=np.int64
        )

    def add(self, feature_array):
        self.gradient_arrays_buffer.append(feature_array.flatten())
        self.gradient_shapes_buffer.append(feature_array.shape)

        if self.gradient_arrays_db is None:
            self.create_datasets()

        if len(self.gradient_arrays_buffer) >= self.buffer_size:
            self._write_buffer(self.gradient_arrays_db, self.gradient_arrays_buffer)
            self._write_buffer(self.gradient_shapes_db, self.gradient_shapes_buffer)

            # increment index
            self.idxs['index'] += len(self.gradient_arrays_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.gradient_arrays_buffer = []
        self.gradient_shapes_buffer = []


def perceptual_gradient_computation(im_file, img_shape, edges_info, g_energies, ground_distance, outdir):
    num_cores = multiprocessing.cpu_count()
    rows, cols, channels = img_shape
    edges_index, neighbors_edges = edges_info

    print('Image:', im_file)
    g_energies_lum = g_energies[:, :, 0]
    g_energies_cr = g_energies[:, :, 1]
    g_energies_ci = g_energies[:, :, 2]

    ''' Updating edges weights with similarity measure (OT/KL) '''
    weights_lum = np.array(Parallel(n_jobs=int(num_cores/3))
                           (delayed(em_dist_mine)(np.float64(g_energies_lum[e]), ground_distance) for e in edges_index))

    weights_cr = np.array(Parallel(n_jobs=int(num_cores/3))
                          (delayed(em_dist_mine)(np.float64(g_energies_cr[e]), ground_distance) for e in edges_index))

    weights_ci = np.array(Parallel(n_jobs=int(num_cores/3))
                          (delayed(em_dist_mine)(np.float64(g_energies_ci[e]), ground_distance) for e in edges_index))

    ''' Computing target gradient from the ground truth'''
    ground_truth_segments = np.array(get_segment_from_filename(im_file))
    weights_gt = np.zeros(len(edges_index), dtype=np.float32)
    for truth in ground_truth_segments:
        truth = truth.reshape(rows * cols)
        weights_gt += np.array(Parallel(n_jobs=num_cores)(
            delayed(dist_label)((truth[e[0]], truth[e[1]])) for e in list(edges_index)))

    weights_gt = (weights_gt - min(weights_gt)) / (max(weights_gt) - min(weights_gt))

    stacked_gradients = np.column_stack((weights_lum, weights_cr, weights_ci, weights_gt))

    gradient_lum = np.empty((rows * cols), dtype=np.float32)
    gradient_cr = np.empty((rows * cols), dtype=np.float32)
    gradient_ci = np.empty((rows * cols), dtype=np.float32)
    gradient_gt = np.empty((rows * cols), dtype=np.float32)

    for pp in range(rows * cols):
        gradient_lum[pp] = np.max(weights_lum[neighbors_edges[pp]])
        gradient_cr[pp] = np.max(weights_cr[neighbors_edges[pp]])
        gradient_ci[pp] = np.max(weights_ci[neighbors_edges[pp]])
        gradient_gt[pp] = np.max(weights_gt[neighbors_edges[pp]])

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_lum.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Luminance gradient', fontsize=10)
    plt.savefig(outdir + im_file + '_lum_grad.png')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_cr.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Chrominance (Re) gradient', fontsize=10)
    plt.savefig(outdir + im_file + '_cr_grad.png')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_ci.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Chrominance (Im) gradient', fontsize=10)
    plt.savefig(outdir + im_file + '_ci_grad.png')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_gt.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Ground truth gradient', fontsize=10)
    plt.savefig(outdir + im_file + '_gt_grad.png')

    plt.cla()
    plt.clf()
    plt.close('all')

    return stacked_gradients


def get_gt_min_nsegments(segments):
    n_labels = []
    for truth in segments:
        n_labels.append(len(np.unique(truth)))
    min_nseg = min(n_labels)
    pos_min_nseg = n_labels.index(min_nseg)

    return segments[pos_min_nseg]


def prepare_dataset(img_id, image, gabor_features, img_shape):
    ground_truth = np.array(get_segment_from_filename(img_id))
    min_ground_truth = get_gt_min_nsegments(ground_truth)
    edges_index, neighbors_edges = get_pixel_graph(kneighbors, img_shape)

    return img_id, image, img_shape, gabor_features, min_ground_truth, edges_index, neighbors_edges


def generate_h5_graph_gradients_dataset(num_imgs, kneighbors, similarity_measure):
    num_cores = -1
    hdf5_indir_im = Path('../../data/hdf5_datasets/'+str(num_imgs)+'images/' + 'images')
    hdf5_indir_feat = Path('../../data/hdf5_datasets/'+str(num_imgs)+'images/' + 'features')
    hdf5_outdir = Path('../../data/hdf5_datasets/'+str(num_imgs)+'images/' + 'gradients/'+str(kneighbors) + 'nn_' + similarity_measure)

    num_imgs_dir = str(num_imgs)+'images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = images_file["images"][:]
    img_shapes = images_file["image_shapes"][:]
    img_ids = images_file["image_ids"][:]

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    ''' Computing Graphs for all images'''
    edges_and_neighbors = Parallel(n_jobs=num_cores)(
        delayed(get_pixel_graph)(kneighbors, img_shape) for img_shape in img_shapes)

    feat_dirs = sorted(os.listdir(hdf5_indir_feat))
    for features_input_dir in feat_dirs:
        time_start = time.time()
        with h5py.File(hdf5_indir_feat / features_input_dir / 'Gabor_features.h5', "r+") as features_file:
            print('Reading Berkeley features data set')
            print('Gabor configuration: ', features_input_dir)
            t0 = time.time()
            feature_vectors = features_file["gabor_features"][:]

            n_freq = features_file.attrs['num_freq']
            n_angles = features_file.attrs['num_angles']
            fb = features_file.attrs['frequency_bandwidth']

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0] * shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            # Compute ground distance matrix
            ground_distance = cost_matrix_texture(n_freq, n_angles, fb)

            outdir = '../outdir/' + \
                     num_imgs_dir + \
                     'gradients/' + \
                     (str(kneighbors) + 'nn_' + similarity_measure) + '/' + \
                     features_input_dir + '/'

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            perceptual_gradients = Parallel(n_jobs=num_cores)(delayed(
                perceptual_gradient_computation)(im_file, img_shape, edges_info, g_energies, ground_distance, outdir)
                                                              for im_file, img, img_shape, edges_info, g_energies in
                                                              zip(img_ids, images, img_shapes, edges_and_neighbors,
                                                                  gabor_features_norm))

            h5_outdir = hdf5_outdir / features_input_dir
            h5_outdir.mkdir(parents=True, exist_ok=True)

            with ImageIndexer(h5_outdir / 'gradients.h5',
                              buffer_size=num_imgs,
                              num_of_images=num_imgs) as imageindexer:

                for gradients in perceptual_gradients:
                    imageindexer.add(gradients)

        time_end = time.time()
        print('Gradient computation time: %.2fs' % (time_end - time_start))


if __name__ == '__main__':
    num_imgs = 7

    # Graph function parameters
    kneighbors = 4  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Graph distance parameters
    method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    generate_h5_graph_gradients_dataset(num_imgs, kneighbors, method)
