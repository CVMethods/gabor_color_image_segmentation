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
from source.plot_save_figures import *


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


def perceptual_gradient_computation(im_file, img, regions_slic, graph_raw, g_energies, outdir):
    print('##############################', im_file, '##############################')

    graph_lum = graph_raw.copy()
    graph_cr = graph_raw.copy()
    graph_ci = graph_raw.copy()
    graph_gt = graph_raw.copy()

    ''' Updating edges weights with similarity measure (OT/KL) '''
    graph_weighted_lum = update_edges_weight(regions_slic, graph_lum, g_energies[:, :, :, 0], ground_distance, method)
    graph_weighted_cr = update_edges_weight(regions_slic, graph_cr, g_energies[:, :, :, 1], ground_distance, method)
    graph_weighted_ci = update_edges_weight(regions_slic, graph_ci, g_energies[:, :, :, 2], ground_distance, method)

    weights_lum = np.fromiter(nx.get_edge_attributes(graph_weighted_lum, 'weight').values(), dtype=np.float32)
    weights_cr = np.fromiter(nx.get_edge_attributes(graph_weighted_cr, 'weight').values(), dtype=np.float32)
    weights_ci = np.fromiter(nx.get_edge_attributes(graph_weighted_ci, 'weight').values(), dtype=np.float32)

    ''' Computing target gradient from the ground truth'''
    ground_truth_segments = np.array(get_segment_from_filename(im_file))
    for truth in ground_truth_segments:
        graph_gt = update_groundtruth_edges_weight(regions_slic, graph_gt, truth)
        graph_weighted_gt = graph_gt.copy()

    weights_gt = np.fromiter(nx.get_edge_attributes(graph_weighted_gt, 'weight').values(), dtype=np.float32)
    weights_gt = (weights_gt - min(weights_gt)) / (max(weights_gt) - min(weights_gt))

    for i_edge, e in enumerate(list(graph_weighted_gt.edges)):
        graph_weighted_gt[e[0]][e[1]]['weight'] = weights_gt[i_edge]

    stacked_gradients = np.column_stack((weights_lum, weights_cr, weights_ci, weights_gt))

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    # General Params
    save_fig = True
    fontsize = 10
    file_name = im_file

    # Show Graph with updated weights
    fig_title = 'Lum Weighted Graph (' + graph_type + ')'
    img_name = '_weighted_lum_' + graph_type
    colbar_lim = (min(weights_lum), max(weights_lum))
    show_and_save_imgraph(img, regions_slic, graph_weighted_lum, fig_title, img_name, fontsize, save_fig, outdir,
                          file_name, colbar_lim)

    fig_title = 'Cr Weighted Graph (' + graph_type + ')'
    img_name = '_weighted_cr_' + graph_type
    colbar_lim = (min(weights_cr), max(weights_cr))
    show_and_save_imgraph(img, regions_slic, graph_weighted_cr, fig_title, img_name, fontsize, save_fig,
                          outdir, file_name, colbar_lim)

    fig_title = 'Ci Weighted Graph (' + graph_type + ')'
    img_name = '_weighted_ci_' + graph_type
    colbar_lim = (min(weights_ci), max(weights_ci))
    show_and_save_imgraph(img, regions_slic, graph_weighted_ci, fig_title, img_name, fontsize, save_fig,
                          outdir, file_name, colbar_lim)

    fig_title = 'Gt Weighted Graph (' + graph_type + ')'
    img_name = '_weighted_gt_' + graph_type
    colbar_lim = (min(weights_gt), max(weights_gt))
    show_and_save_imgraph(img, regions_slic, graph_weighted_gt, fig_title, img_name, fontsize, save_fig,
                          outdir, file_name, colbar_lim)

    return stacked_gradients


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_spix = hdf5_dir / 'complete' / 'superpixels'
        hdf5_indir_feat = hdf5_dir / 'complete' / 'features'
        hdf5_outdir = hdf5_dir / 'complete' / 'gradients'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_spix = hdf5_dir / '7images' / 'superpixels'
        hdf5_indir_feat = hdf5_dir / '7images/' / 'features'
        hdf5_outdir = hdf5_dir / '7images' / 'gradients'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_spix = hdf5_dir / '25images' / 'superpixels'
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

    superpixels_file = h5py.File(hdf5_indir_spix / "Berkeley_superpixels.h5", "r+")
    superpixels_vectors = np.array(superpixels_file["/superpixels"])

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    superpixels = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img_spix, (shape[0], shape[1])) for img_spix, shape in
        zip(superpixels_vectors, img_shapes)))

    n_regions = superpixels_file.attrs['num_slic_regions']

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    ''' Computing Graphs for all images'''
    # Graph function parameters
    graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
    kneighbors = 8
    radius = 10

    all_raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type, kneighbors, radius) for img, regions_slic in zip(images, superpixels))

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    # General Params
    save_fig = True
    fontsize = 10

    outdir = '../outdir/perceptual_gradient/' + \
             num_imgs_dir + 'slic_level/' + \
             '/computation_support/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Show Input images
    fig_title = 'Input Image'
    Parallel(n_jobs=num_cores)(
            delayed(show_and_save_img)(img, fig_title, fontsize, save_fig, outdir, file_name)
            for img, file_name in zip(images, img_ids))

    # Show SLIC results
    fig_title = 'Superpixel Regions'
    img_name = '_slic'
    Parallel(n_jobs=num_cores)(
        delayed(show_and_save_regions)(img, regions_slic, fig_title, img_name, fontsize, save_fig, outdir, file_name)
        for img, regions_slic, file_name in zip(images, superpixels, img_ids))

    # Show Graphs with uniform weight
    fig_title = 'Graph (' + graph_type + ')'
    img_name = '_raw_' + graph_type
    colbar_lim = (0, 1)
    Parallel(n_jobs=num_cores)(
        delayed(show_and_save_imgraph)(img, regions_slic, graph_raw, fig_title, img_name, fontsize, save_fig, outdir, file_name,
                              colbar_lim) for img, regions_slic, graph_raw, file_name in zip(images, superpixels, all_raw_graphs, img_ids))

    # Graph distance parameters
    method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    input_files = os.listdir(hdf5_indir_feat)
    for features_input_file in input_files:
        time_start = time.time()
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = np.array(features_file["/gabor_features"])
            feature_shapes = np.array(features_file["/feature_shapes"])

            n_freq = features_file.attrs['num_freq']
            n_angles = features_file.attrs['num_angles']

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            # Compute ground distance matrix
            ground_distance = cost_matrix_texture(n_freq, n_angles)

            output_file_name = features_input_file.split('_')
            output_file_name[1] = 'PerceptualGradients'
            output_file = '_'.join(output_file_name)

            outdir = '../outdir/perceptual_gradient/' + \
                     num_imgs_dir + \
                     'slic_level/' \
                     + output_file[:-3] + '/' + \
                     str(n_regions) + '_regions/'

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            perceptual_gradients = Parallel(n_jobs=num_cores)(delayed(
                perceptual_gradient_computation)(im_file, img, regions_slic, graph_raw, g_energies, outdir) for im_file, img, regions_slic, graph_raw, g_energies in
                zip(img_ids, images, superpixels, all_raw_graphs, gabor_features_norm))

            with ImageIndexer(hdf5_outdir / output_file,
                              buffer_size=num_imgs,
                              num_of_images=num_imgs) as imageindexer:

                for gradients in perceptual_gradients:
                    imageindexer.add(gradients)
                    imageindexer.db.attrs['num_slic_regions'] = n_regions

        time_end = time.time()
        print('Gradient computation time: %.2fs' % (time_end - time_start))