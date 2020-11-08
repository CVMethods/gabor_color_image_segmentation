import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd

from pathlib import Path
from joblib import Parallel, delayed
from sklearn.neighbors import kneighbors_graph

sys.path.append('../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *
from source.plot_save_figures import *

def image_hierachical_watershed_segmentation(im_file, img_shape, image_contours, outdir):
    rows, cols, channels = img_shape

    image_contours = image_contours.astype(np.float32)
    graph = hg.get_4_adjacency_graph((rows, cols))
    edge_weights = hg.weight_graph(graph, image_contours, hg.WeightFunction.mean)
    tree, altitudes = hg.watershed_hierarchy_by_volume(graph, edge_weights)

    # we construct a sketch of the saliency map just for illustration
    sm = hg.graph_4_adjacency_2_khalimsky(graph, hg.saliency(tree, altitudes))**0.5
    sm = np.pad(sm, ((0, 2), (0, 2)), mode='edge')
    sm /= np.max(sm)
    mdic = {"ucm2": sm}
    savemat(outdir + im_file + '.mat', mdic)

    outdir_cont = outdir.replace(outdir.split('/')[3], 'image_contours_higra')
    if not os.path.exists(outdir_cont):
        os.makedirs(outdir_cont)

    img_grad = np.uint8(sm[1::2, 1::2] * 255)
    ##############################################################################
    '''Visualization Section: show and/or save images'''
    img = Image.fromarray(img_grad)
    img.save(outdir_cont + im_file + '.png')

    # plt.figure(dpi=180)
    # plt.imshow(sm[1::2, 1::2], cmap='gray')
    # plt.show(block=False)
    # pdb.set_trace()


def generate_hwatershed_segmentations_from_imgcontours(num_imgs, similarity_measure, kneighbors=None, n_slic=None,
                                                       graph_type=None):
    num_cores = -1

    if kneighbors is not None:
        slic_level = False
        pixel_level = True
        img_cont_indir = str(kneighbors) + 'nn_' + similarity_measure

    elif n_slic is not None and graph_type is not None:
        slic_level = True
        pixel_level = False
        img_cont_indir = str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure



    hdf5_indir_im = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_cont = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'image_contours/' + img_cont_indir)

    num_imgs_dir = str(num_imgs) + 'images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = images_file["images"][:]
    img_shapes = images_file["image_shapes"][:]
    img_ids = images_file["image_ids"][:]
    img_subdirs = images_file["image_subdirs"][:]

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    test_indices = []
    for ii in range(len(images)):
        if img_subdirs[ii] == 'test':
            test_indices.append(ii)

    img_ids = img_ids[test_indices]
    images = images[test_indices]
    img_shapes = img_shapes[test_indices]

    model_input_dirs = sorted(os.listdir(hdf5_indir_cont))
    for mm, model_name in enumerate(model_input_dirs):
        input_directories = sorted(os.listdir(hdf5_indir_cont / model_name))

        for gradients_input_dir in input_directories:
            with h5py.File(hdf5_indir_cont / model_name / gradients_input_dir / 'image_contours.h5',
                           "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_dir)

                imgs_contours = gradients_file["image_contours"][:]

                outdir = '../outdir/' + \
                         num_imgs_dir + \
                         'image_segmentations/' + \
                         img_cont_indir + '/' + \
                         model_name + '/' + \
                         gradients_input_dir + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                img_segmentations = Parallel(n_jobs=-1)(
                            delayed(image_hierachical_watershed_segmentation)(im_file, img_shape, image_contours, outdir)
                            for im_file, img_shape, image_contours in
                            zip(img_ids, img_shapes, imgs_contours))

if __name__ == '__main__':
    num_imgs = 500

    kneighbors = 4

    n_slic = 500 * 11

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    # generate_hwatershed_segmentations_from_imgcontours(num_imgs, similarity_measure, kneighbors=kneighbors, n_slic=None,
    #                                  graph_type=None)
    #
    generate_hwatershed_segmentations_from_imgcontours(num_imgs, similarity_measure, kneighbors=None, n_slic=n_slic,
                                                       graph_type=graph_type)
