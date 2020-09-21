# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys

sys.path.append('../')

import h5py
import time
import itertools

from pathlib import Path
from joblib import Parallel, delayed

from source.myGaborFunctions import *
from source.color_transformations import *


class ImageIndexer(object):
    def __init__(self, db_path, buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.feature_arrays_db = None
        self.feature_shapes_db = None
        self.idxs = {"index": 0}

        self.feature_arrays_buffer = []
        self.feature_shapes_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.feature_arrays_buffer:
            print("writing last buffers")
            print(len(self.feature_arrays_buffer))

            self._write_buffer(self.feature_arrays_db, self.feature_arrays_buffer)
            self._write_buffer(self.feature_shapes_db, self.feature_shapes_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.feature_arrays_db = self.db.create_dataset(
            "gabor_features",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('float32'))
        )

        self.feature_shapes_db = self.db.create_dataset(
            "feature_shapes",
            shape=(self.num_of_images, 2),
            maxshape=(None, 2),
            dtype=np.int64
        )

    def add(self, feature_array):
        self.feature_arrays_buffer.append(feature_array.flatten())
        self.feature_shapes_buffer.append(feature_array.shape)

        if self.feature_arrays_db is None:
            self.create_datasets()

        if len(self.feature_arrays_buffer) >= self.buffer_size:
            self._write_buffer(self.feature_arrays_db, self.feature_arrays_buffer)
            self._write_buffer(self.feature_shapes_db, self.feature_shapes_buffer)

            # increment index
            self.idxs['index'] += len(self.feature_arrays_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.feature_arrays_buffer = []
        self.feature_shapes_buffer = []


def get_gabor_features(img_complex, gabor_filters, r_type, gsmooth, opn, selem_size):
    num_cores = -1
    channels, rows, cols = img_complex.shape
    filter_responses = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(applyGabor_filterbank)(img_channel, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                       morph_opening=opn, se_z=selem_size) for img_channel in img_complex))

    g_responses_norm = filter_responses
    # g_responses_norm = normalize_img(filter_responses, rows, cols)
    # print(np.sum(g_responses_norm**2) / (rows*cols))

    g_features = []
    for ff in range(g_responses_norm.shape[1]):
        g_features.append(reshape4clustering(g_responses_norm[0][ff], rows, cols))
        g_features.append(reshape4clustering(g_responses_norm[1][ff], rows, cols))
        g_features.append(reshape4clustering(g_responses_norm[2][ff], rows, cols))

    g_features = np.array(g_features).T

    return g_features


def generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations):
    hdf5_indir = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/images')
    hdf5_outdir = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/features')

    # Generating Gabor filterbank
    r_type = 'L2'  # 'real'
    gsmooth = False
    opn = True
    selem_size = 2
    num_cores = -1

    for (min_period, max_period), (fb, ab), (c1, c2), stds in itertools.product(periods, bandwidths,
                                                                              crossing_points, deviations):
        print('Generating Gabor filterbank')

        gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

        n_freq = len(frequencies)
        n_angles = len(angles)
        color_space = 'LAB'  # HS, HSV, HV, LAB

        # Read hdf5 file and extract its information
        print('Reading Berkeley image data set')
        t0 = time.time()
        file = h5py.File(hdf5_indir / "Berkeley_images.h5", "r+")
        image_vectors = np.array(file["/images"])
        img_shapes = np.array(file["/image_shapes"])

        images = Parallel(n_jobs=num_cores)(
            delayed(np.reshape)(img, shape) for img, shape in zip(image_vectors, img_shapes))

        t1 = time.time()
        print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

        # ## Parallel computation of Gabor features
        print('Computing Gabor features:')
        t0 = time.time()
        twoChannel_imgs = Parallel(n_jobs=num_cores)(
            delayed(img2complex_normalized_colorspace)(img, shape, color_space) for img, shape in
            zip(images, img_shapes))

        gabor_features = Parallel(n_jobs=num_cores, prefer='processes')(
            delayed(get_gabor_features)(img, gabor_filters, r_type, gsmooth, opn, selem_size) for img in
            twoChannel_imgs)
        t1 = time.time()
        print('Features computing time (using Parallel joblib): %.2fs' % (t1 - t0))

        gabor_conf = ('%df_%da_%dp_%dp_%.2ffb_%dab_%.2fcpf_%.2fcpa_%.1fstds'
                                     % (n_freq, n_angles, min_period, max_period, fb, ab, c1, c2, stds))

        h5_outdir = hdf5_outdir / gabor_conf
        h5_outdir.mkdir(parents=True, exist_ok=True)

        with ImageIndexer(h5_outdir / 'Gabor_features.h5',
                          buffer_size=num_imgs,
                          num_of_images=num_imgs) as imageindexer:

            for features in gabor_features:
                imageindexer.add(features)
                imageindexer.db.attrs['num_freq'] = n_freq
                imageindexer.db.attrs['num_angles'] = n_angles
                imageindexer.db.attrs['min_period'] = min_period
                imageindexer.db.attrs['max_period'] = max_period
                imageindexer.db.attrs['frequency_bandwidth'] = fb
                imageindexer.db.attrs['angular_bandwidth'] = ab
                imageindexer.db.attrs['crossing_point1'] = c1
                imageindexer.db.attrs['crossing_point2'] = c2
                imageindexer.db.attrs['num_stds'] = stds


if __name__ == '__main__':
    num_imgs = 7

    periods = [(2., 25.), (2., 45.)]
    bandwidths = [(0.7, 30), (1.0, 45)]
    crossing_points = [(0.75, 0.75), (0.9, 0.9), (0.9, 0.75), (0.75, 0.9)]
    deviations = [3.0]
    generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations)
