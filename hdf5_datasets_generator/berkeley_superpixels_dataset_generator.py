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
        self.superpixels_db = None
        self.idxs = {"index": 0}

        self.superpixels_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.superpixels_buffer:
            print("writing last buffers")
            print(len(self.superpixels_buffer))

            self._write_buffer(self.superpixels_db, self.superpixels_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.superpixels_db = self.db.create_dataset(
            "superpixels",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('int64'))
        )

    def add(self, feature_array):
        self.superpixels_buffer.append(feature_array.flatten())

        if self.superpixels_db is None:
            self.create_datasets()

        if len(self.superpixels_buffer) >= self.buffer_size:
            self._write_buffer(self.superpixels_db, self.superpixels_buffer)

            # increment index
            self.idxs['index'] += len(self.superpixels_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.superpixels_buffer = []


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_outdir = hdf5_dir / 'complete' / 'superpixels'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_outdir = hdf5_dir / '7images' / 'superpixels'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_outdir = hdf5_dir / '25images' / 'superpixels'
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

    ''' Computing superpixel regions for all images'''
    # Superpixels function parameters
    n_regions = 500 * 4
    convert2lab = True

    superpixels = Parallel(n_jobs=num_cores)(
        delayed(slic_superpixel)(img, n_regions, convert2lab) for img in images)


    with ImageIndexer(hdf5_outdir / "Berkeley_superpixels.h5",
                      buffer_size=num_imgs,
                      num_of_images=num_imgs) as imageindexer:

        for slic in superpixels:
            imageindexer.add(slic)
            imageindexer.db.attrs['num_slic_regions'] = n_regions

