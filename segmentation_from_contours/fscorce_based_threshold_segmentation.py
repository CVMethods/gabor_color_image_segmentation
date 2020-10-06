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

def image_threshold_segmentation(im_file, img_shape, thr_info, image_contours, outdir):
    rows, cols, channels = img_shape
    image_contours = (image_contours.reshape(rows, cols)/255.) > thr_info

    plt.figure(dpi=180)
    plt.imshow(image_contours, cmap='gray')
    plt.show(block=False)
    pdb.set_trace()


def generate_segmentations_from_imgcontours(num_imgs, similarity_measure, gradients_dir, kneighbors=None, n_slic=None,
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
    hdf5_indir_grad = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/predicted_gradients/' + img_cont_indir)
    hdf5_indir_cont = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'image_contours/' + img_cont_indir)

    num_imgs_dir = str(num_imgs) + 'images/'

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

    test_indices = []
    for ii in range(len(images)):
        if img_subdirs[ii] == 'test':
            test_indices.append(ii)

    img_ids = img_ids[test_indices]
    images = images[test_indices]
    img_shapes = img_shapes[test_indices]

    model_input_dirs = sorted(os.listdir(hdf5_indir_grad))
    for mm, model_name in enumerate(model_input_dirs):
        input_directories = sorted(os.listdir(hdf5_indir_grad / model_name))

        for gradients_input_dir in input_directories:
            with h5py.File(hdf5_indir_cont / model_name / gradients_input_dir / 'image_contours.h5',
                           "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_dir)

                bdry_thr_indir = '../outdir/' + \
                             num_imgs_dir + \
                             'image_contours/' + \
                             img_cont_indir + '/' + \
                             model_name + '/' + \
                             gradients_input_dir + '/'

                imgs_contours = np.array(gradients_file["/image_contours"])

                print(imgs_contours.shape)
                
                bdry_thr_dataframe = pd.read_csv(bdry_thr_indir + 'eval_bdry_img.txt', delimiter='\s+', header=None)
                bdry_thr_dataframe.columns = ["img index", "best Thr", "recall", "best precision", "best fscore"]
                bdry_thr_array =  bdry_thr_dataframe["best Thr"].to_numpy().flatten()


                outdir = '../outdir/' + \
                         num_imgs_dir + \
                         'image_segmentations/' + \
                         img_cont_indir + '/' + \
                         model_name + '/' + \
                         gradients_input_dir + '/'

                img_segmentations = Parallel(n_jobs=1)(
                            delayed(image_threshold_segmentation)(im_file, img_shape, thr_info, image_contours, outdir)
                            for im_file, img_shape, thr_info, image_contours in
                            zip(img_ids, img_shapes, bdry_thr_array, imgs_contours))

if __name__ == '__main__':
    num_imgs = 7

    kneighbors = 4

    n_slic = 500 * 4

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    gradients_dir = 'predicted_gradients'  # 'predicted_gradients'

    # generate_segmentations_from_imgcontours(num_imgs, similarity_measure, gradients_dir, kneighbors=kneighbors, n_slic=None,
    #                                  graph_type=None)

    generate_segmentations_from_imgcontours(num_imgs, similarity_measure, gradients_dir, kneighbors=None, n_slic=n_slic,
                                     graph_type=graph_type)
