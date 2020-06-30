# -*- coding: utf-8 -*-
# !/usr/bin/env python


import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path
from joblib import Parallel, delayed

from BSD_metrics.metrics import *
from Gabor_analysis.myGaborFunctions import *
from complexColor.color_transformations import *

class ImageIndexer(object):
    def __init__(self, db_path, fixed_image_shape=(512, 512, 3), fixed_features_shape=(512, 20), buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.fixed_image_shape = fixed_image_shape
        self.fixed_features_shape = fixed_features_shape
        self.image_vector_db = None
        self.feature_vector_db = None
        self.idxs = {"index": 0}

        self.image_vector_buffer = []
        self.feature_vector_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.feature_vector_buffer:
            print("writing last buffers")
            print(len(self.feature_vector_buffer))

            self._write_buffer(self.feature_vector_db, self.feature_vector_buffer)
            self._write_buffer(self.image_vector_db, self.image_vector_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took {0}s".format(time.time() - self.t0))

    @property
    def image_vector_size(self):
        if self.fixed_image_shape:
            return self.fixed_image_shape[2], self.fixed_image_shape[0]*self.fixed_image_shape[1]
        else:
            return None

    def create_datasets(self):
        IMG_ROWS, IMG_COLS, CHANN = self.fixed_image_shape
        IMG_PXLS, IMG_FEAT = self.fixed_features_shape

        self.image_vector_db = self.db.create_dataset(
            "complex_images",
            shape=(self.num_of_images, 3, IMG_ROWS * IMG_COLS),
            maxshape=(self.num_of_images, None, None),
            dtype=np.float32,
            chunks=True
        )

        self.feature_vector_db = self.db.create_dataset(
            "gabor_features",
            shape=(self.num_of_images, IMG_PXLS, IMG_FEAT),
            maxshape=(self.num_of_images, None, None),
            dtype=np.float32,
            chunks=True
        )

    def add(self, image_vector, feature_vector):
        self.image_vector_buffer.append(image_vector.reshape(self.image_vector_size))
        self.feature_vector_buffer.append(feature_vector)

        if None in (self.image_vector_db, self.feature_vector_db):
            self.create_datasets()

        if len(self.feature_vector_buffer) >= self.buffer_size:
            self._write_buffer(self.image_vector_db, self.image_vector_buffer)
            self._write_buffer(self.feature_vector_db, self.feature_vector_buffer)

            # increment index
            self.idxs['index'] += len(self.image_vector_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.image_vector_buffer = []
        self.feature_vector_buffer = []


def get_gabor_features(img_complex, gabor_filters, r_type, gsmooth, opn, selem_size):
    channels, rows, cols = img_complex.shape
    filter_responses = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(applyGabor_filterbank)(img_channel, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                       morph_opening=opn, se_z=selem_size) for img_channel in img_complex))

    g_responses_norm = normalize_img(filter_responses, rows, cols)
    # print(np.sum(g_responses_norm**2) / (rows*cols))

    features = []
    for ff in range(g_responses_norm.shape[1]):
        features.append(reshape4clustering(g_responses_norm[0][ff], rows, cols))
        features.append(reshape4clustering(g_responses_norm[1][ff], rows, cols))
        features.append(reshape4clustering(g_responses_norm[2][ff], rows, cols))

    features = np.array(features).T

    return features


if __name__ == '__main__':

    num_imgs = 500

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_dir = Path('../data/hdf5_datasets/complete/')

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_dir = Path('../data/hdf5_datasets/7images/')

    # Generating Gabor filterbank
    min_period = 2.
    max_period = 35.
    fb = 1  # 0.7  #
    ab = 45  # 30  #
    c1 = 0.65
    c2 = 0.65
    stds = 3.5
    print('Generating Gabor filterbank')
    gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

    n_freq = len(frequencies)
    n_angles = len(angles)
    color_space = 'HS'
    color_space = 'HS'

    r_type = 'L2'  # 'real'
    gsmooth = True
    opn = True
    selem_size = 1
    num_cores = -1

    # Read hdf5 file and extract its information
    print('Reading Berkeley image data set')
    file = h5py.File(hdf5_dir / "Berkeley_images.h5", "r+")
    image_vectors = np.array(file["/images"])
    img_shapes = np.array(file["/image_shapes"])
    img_ids = np.array(file["/image_ids"])
    num_segments = np.array(file["/num_seg"])

    images = Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes))

    # ## Parallel computation of Gabor features
    print('Computing Gabor features:')
    t0 = time.time()
    twoChannel_imgs = Parallel(n_jobs=num_cores)(
        delayed(img2complex_normalized_colorspace)(img, shape, color_space) for img, shape in zip(images, img_shapes))

    gabor_features = Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(get_gabor_features)(img, gabor_filters, r_type, gsmooth, opn, selem_size) for img in twoChannel_imgs)
    t1 = time.time()
    print('Features computing time (using Parallel joblib): %.2fs' % (t1 - t0))

    with ImageIndexer(hdf5_dir / "Berkeley_GaborFeatures.h5",
                      fixed_image_shape=(481, 321, 3),
                      fixed_features_shape=(img_shapes[0][0]*img_shapes[0][1], n_freq*n_angles*img_shapes[0][2]),
                      buffer_size=num_imgs,
                      num_of_images=num_imgs) as imageindexer:

        for img, features in zip(twoChannel_imgs, gabor_features):
            imageindexer.add(img, features)
        # Parallel(n_jobs=num_cores)(delayed(imageindexer.add)(img, features) for img, features in zip(twoChannel_imgs, gabor_features))

    # # Read hdf5 file and extract its information
    # features_file = h5py.File(hdf5_dir/ "Berkeley_GaborFeatures.h5", "r+")
    # complex_images = np.array(features_file["/complex_images"])
    # features = np.array(features_file["/gabor_features"])
    #
    # pdb.set_trace()
