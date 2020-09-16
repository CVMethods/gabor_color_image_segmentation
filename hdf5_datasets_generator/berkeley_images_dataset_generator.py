import os
import time
import h5py
import pdb
import numpy as np

from pathlib import Path
from skimage import io


class ImageIndexer(object):
    def __init__(self, db_path, buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.idxs = {"index": 0}

        self.image_ids_db = None
        self.image_subdirs_db = None
        self.image_arrays_db = None
        self.image_shapes_db = None

        self.image_ids_buffer = []
        self.image_subdirs_buffer = []
        self.image_arrays_buffer = []
        self.image_shapes_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_ids_buffer:
            print("writing last buffers")
            print(len(self.image_ids_buffer))

            self._write_buffer(self.image_ids_db, self.image_ids_buffer)
            self._write_buffer(self.image_subdirs_db, self.image_subdirs_buffer)
            self._write_buffer(self.image_arrays_db, self.image_arrays_buffer)
            self._write_buffer(self.image_shapes_db, self.image_shapes_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.image_ids_db = self.db.create_dataset(
            "image_ids",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str)
        )

        self.image_subdirs_db = self.db.create_dataset(
            "image_subdirs",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str)
        )

        self.image_arrays_db = self.db.create_dataset(
            "images",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('uint8'))
        )

        self.image_shapes_db = self.db.create_dataset(
            "image_shapes",
            shape=(self.num_of_images,  3),
            dtype=np.int64
        )

    def add(self, image_name, image_subdir, image_array):
        self.image_ids_buffer.append(image_name)
        self.image_subdirs_buffer.append(image_subdir)
        self.image_arrays_buffer.append(image_array.flatten())
        self.image_shapes_buffer.append(image_array.shape)

        if self.image_ids_db is None:
            self.create_datasets()

        if len(self.image_ids_buffer) >= self.buffer_size:
            self._write_buffer(self.image_ids_db, self.image_ids_buffer)
            self._write_buffer(self.image_subdirs_db, self.image_subdirs_buffer)
            self._write_buffer(self.image_arrays_db, self.image_arrays_buffer)
            self._write_buffer(self.image_shapes_db, np.array(self.image_shapes_buffer))

            # increment index
            self.idxs['index'] += len(self.image_ids_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.image_ids_buffer = []
        self.image_subdirs_buffer = []
        self.image_arrays_buffer = []
        self.image_shapes_buffer = []


def generate_h5_images_dataset(num_imgs):
    bsd_path = Path('../../data/images/' + str(num_imgs) + 'images')
    hdf5_outdir = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/images')

    hdf5_outdir.mkdir(parents=True, exist_ok=True)

    with ImageIndexer(hdf5_outdir / "Berkeley_images.h5",
                      buffer_size=num_imgs,
                      num_of_images=num_imgs) as imageindexer:

        subdirectories = os.listdir(bsd_path)

        for subdir in subdirectories:
            imgs_path = bsd_path / subdir
            list_imgs = os.listdir(imgs_path)
            for file_name in list_imgs:
                image_array = io.imread(imgs_path / file_name)
                imageindexer.add(file_name[:-4], subdir, image_array)


if __name__ == '__main__':
    num_imgs = 500
    generate_h5_images_dataset(num_imgs)
