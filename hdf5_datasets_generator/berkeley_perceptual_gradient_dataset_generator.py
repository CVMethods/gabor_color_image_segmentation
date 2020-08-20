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


def compute_perceptual_gradient(i_dataset, dataset):
    img_id, image, img_size, g_energies, ground_truth_segmnt, edges_index, neighbors_edges = dataset
    rows, cols, channels = img_size

    print('dataset image:', img_id)

    g_energies_lum = g_energies[:, :, 0]
    g_energies_cr = g_energies[:, :, 1]
    g_energies_ci = g_energies[:, :, 2]

    weights_lum = np.array(Parallel(n_jobs=-1)
                            (delayed(em_dist_mine)(np.float64(g_energies_lum[e]), ground_distance) for e in edges_index))

    weights_cr = np.array(Parallel(n_jobs=-1)
                           (delayed(em_dist_mine)(np.float64(g_energies_cr[e]), ground_distance) for e in edges_index))

    weights_ci = np.array(Parallel(n_jobs=-1)
                           (delayed(em_dist_mine)(np.float64(g_energies_ci[e]), ground_distance) for e in edges_index))

    gradient_lum = np.empty((rows * cols), dtype=np.float32)
    gradient_cr = np.empty((rows * cols), dtype=np.float32)
    gradient_ci = np.empty((rows * cols), dtype=np.float32)

    for pp in range(rows * cols):
        gradient_lum[pp] = np.max(weights_lum[neighbors_edges[pp]])
        gradient_cr[pp] = np.max(weights_cr[neighbors_edges[pp]])
        gradient_ci[pp] = np.max(weights_ci[neighbors_edges[pp]])

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_lum.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Luminance gradient', fontsize=10)
    plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_lum_grad.png')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_cr.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Chrominance (Re) gradient', fontsize=10)
    plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_cr_grad.png')

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.imshow(gradient_ci.reshape((rows, cols)), cmap='gray')
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title('Chrominance (Im) gradient', fontsize=10)
    plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_ci_grad.png')

    plt.cla()
    plt.clf()
    plt.close()

    return np.column_stack((gradient_lum, gradient_cr, gradient_ci))


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


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_feat = hdf5_dir / 'complete' / 'features'
        hdf5_outdir = hdf5_dir / 'complete' / 'gradients'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_feat = hdf5_dir / '7images/' / 'features'
        hdf5_outdir = hdf5_dir / '7images' / 'gradients'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_feat = hdf5_dir / '25images' / 'features'
        hdf5_outdir = hdf5_dir / '25images' / 'gradients'
        num_imgs_dir = '25images/'

    hdf5_outdir.mkdir(parents=True, exist_ok=True)

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = np.array(images_file["/images"])
    img_shapes = np.array(images_file["/image_shapes"])
    img_ids = np.array(images_file["/image_ids"])
    img_subdirs = np.array(images_file["/image_subdirs"])

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    # img_training = images[img_subdirs == 'train']
    # pdb.set_trace()

    # Graph function parameters
    kneighbors = 8

    input_files = os.listdir(hdf5_indir_feat)
    for features_input_file in input_files:
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = np.array(features_file["/gabor_features"])
            feature_shapes = np.array(features_file["/feature_shapes"])

            n_freq = features_file.attrs['num_freq']
            n_angles = features_file.attrs['num_angles']

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0] * shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            # Compute ground distance matrix
            ground_distance = cost_matrix_texture(n_freq, n_angles)

            iterator = zip(img_ids, images, gabor_features_norm, img_shapes)

            datasets = Parallel(n_jobs=num_cores)(
                delayed(prepare_dataset)(im_id, image, feature, shape) for im_id, image, feature, shape in iterator)

            output_file_name = features_input_file.split('_')
            output_file_name[1] = 'PerceptualGradients'
            output_file = '_'.join(output_file_name)

            outdir = '../outdir/perceptual_gradient/' + num_imgs_dir + 'pixel_level/' + output_file[:-3] + '/'

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            perceptual_gradients = Parallel(n_jobs=num_cores, prefer='processes')(delayed(compute_perceptual_gradient)(i_dataset + 1, dataset) for i_dataset, dataset in enumerate(datasets))


            with ImageIndexer(hdf5_outdir / output_file,
                              buffer_size=num_imgs,
                              num_of_images=num_imgs) as imageindexer:

                for gradients in perceptual_gradients:
                    imageindexer.add(gradients)





