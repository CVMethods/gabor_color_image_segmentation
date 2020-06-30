import h5py
import pdb

from pathlib import Path
from skimage import io
from BSD_metrics.groundtruth import *
from scipy.stats import hmean


class ImageIndexer(object):
    def __init__(self, db_path, fixed_image_shape=(512, 512), buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.fixed_image_shape = fixed_image_shape
        self.image_vector_db = None
        self.image_id_db = None
        self.image_shape_db = None
        self.num_groundtruth_regions_db = None
        #         self.db_index = None
        self.idxs = {"index": 0}

        self.image_vector_buffer = []
        self.image_id_buffer = []
        self.image_shape_buffer = []
        self.num_groundtruth_regions_buffer = []

    #         self.db_index_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_id_buffer:
            print("writing last buffers")
            print(len(self.image_id_buffer))

            self._write_buffer(self.image_id_db, self.image_id_buffer)
            self._write_buffer(self.image_vector_db, self.image_vector_buffer)
            self._write_buffer(self.image_shape_db, self.image_shape_buffer)
            self._write_buffer(self.num_groundtruth_regions_db, self.num_groundtruth_regions_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took {0}".format(time.time() - self.t0))

    @property
    def image_vector_size(self):
        if self.fixed_image_shape:
            return self.fixed_image_shape[0] * self.fixed_image_shape[1], self.fixed_image_shape[2]
        else:
            return None

    def create_datasets(self):

        IMG_ROWS, IMG_COLS, CHANN = self.fixed_image_shape

        self.image_id_db = self.db.create_dataset(
            "image_ids",
            (self.num_of_images,),
            maxshape=None,
            dtype=h5py.special_dtype(vlen=str)

        )

        self.image_vector_db = self.db.create_dataset(
            "images",
            shape=(self.num_of_images, IMG_ROWS * IMG_COLS, 3),
            maxshape=(self.num_of_images, None, None),
            dtype=np.uint8
        )

        self.image_shape_db = self.db.create_dataset(
            "image_shapes",
            shape=(self.num_of_images,  3),
            dtype=np.int64
        )

        self.num_groundtruth_regions_db = self.db.create_dataset(
            "num_seg",
            shape=(self.num_of_images,  4),
            dtype=np.int64
        )

    def add(self, image_name, image_vector, image_shape, num_seg):
        self.image_id_buffer.append(image_name)
        self.image_vector_buffer.append(image_vector.reshape(self.image_vector_size))
        self.image_shape_buffer.append(image_shape)
        self.num_groundtruth_regions_buffer.append(num_seg)

        if None in (self.image_vector_db, self.image_id_db, self.image_shape_db, self.num_groundtruth_regions_db):
            self.create_datasets()

        if len(self.image_id_buffer) >= self.buffer_size:
            self._write_buffer(self.image_id_db, self.image_id_buffer)
            self._write_buffer(self.image_vector_db, self.image_vector_buffer)
            self._write_buffer(self.image_shape_db, self.image_shape_buffer)
            self._write_buffer(self.num_groundtruth_regions_db, self.num_groundtruth_regions_buffer)

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
        self.image_id_buffer = []
        self.image_vector_buffer = []
        self.image_shape_buffer = []
        self.num_groundtruth_regions_buffer = []


def get_num_segments(segments):
    n_labels = []
    for truth in segments:
        n_labels.append(len(np.unique(truth)))
    n_labels = np.array(n_labels)

    return np.array((max(n_labels), min(n_labels), int(n_labels.mean()), int(hmean(n_labels))))


if __name__ == '__main__':

    num_imgs = 500

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        bsd_path = Path('../data/Berkeley/')
        hdf5_dir = Path('../data/hdf5_datasets/complete/')
        hdf5_dir.mkdir(parents=True, exist_ok=True)

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        bsd_path = Path('../data/myFavorite_BSDimages/')
        hdf5_dir = Path('../data/hdf5_datasets/7images/')
        hdf5_dir.mkdir(parents=True, exist_ok=True)

    with ImageIndexer(hdf5_dir / "Berkeley_images.h5",
                      fixed_image_shape=(481, 321, 3),
                      buffer_size=num_imgs,
                      num_of_images=num_imgs) as imageindexer:

        subdirectories = os.listdir(bsd_path)

        for subdir in subdirectories:
            imgs_path = bsd_path / subdir
            list_imgs = os.listdir(imgs_path)
            for file_name in list_imgs:
                image_id = file_name[:-4]
                image_array = io.imread(imgs_path / file_name)
                groundtruth_segments = np.array(get_segment_from_filename(file_name[:-4]))
                n_segments = get_num_segments(groundtruth_segments)
                imageindexer.add(image_id, image_array, image_array.shape, n_segments)