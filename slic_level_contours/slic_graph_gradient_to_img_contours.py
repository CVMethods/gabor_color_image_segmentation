import sys
import h5py

from pathlib import Path

sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *


class ImageIndexer(object):
    def __init__(self, db_path, buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.contour_arrays_db = None
        self.idxs = {"index": 0}

        self.contour_arrays_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.contour_arrays_buffer:
            print("writing last buffers")
            print(len(self.contour_arrays_buffer))

            self._write_buffer(self.contour_arrays_db, self.contour_arrays_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.contour_arrays_db = self.db.create_dataset(
            "image_contours",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('uint8'))
        )

    def add(self, feature_array):
        self.contour_arrays_buffer.append(feature_array.flatten())

        if self.contour_arrays_db is None:
            self.create_datasets()

        if len(self.contour_arrays_buffer) >= self.buffer_size:
            self._write_buffer(self.contour_arrays_db, self.contour_arrays_buffer)

            # increment index
            self.idxs['index'] += len(self.contour_arrays_buffer)

            # clean buffers
            self._clean_buffers()

    def _write_buffer(self, dataset, buf):
        print("Writing buffer {}".format(dataset))
        start = self.idxs['index']
        end = len(buf)
        dataset[start:start + end] = buf

    def _clean_buffers(self):
        self.contour_arrays_buffer = []


def get_img_contours(im_file, img_shape, edges_info, perceptual_gradients, gradients_dir, outdir):
    print('##############################', im_file, '##############################')

    if gradients_dir == 'gradients':
        perceptual_gradients = np.sum(perceptual_gradients[:, :-1], axis=-1)

    rows, cols, channels = img_shape
    edges_index, neighbors_edges = edges_info

    gradient_pred = np.empty((rows * cols), dtype=np.float32)

    for pp in range(rows * cols):
        gradient_pred[pp] = np.max(perceptual_gradients[neighbors_edges[pp]])

    img_grad = (gradient_pred - min(gradient_pred)) / (max(gradient_pred) - min(gradient_pred)) * 255
    img_grad = np.uint8(img_grad.reshape(rows, cols))
    ##############################################################################
    '''Visualization Section: show and/or save images'''
    img = Image.fromarray(img_grad)
    img.save(outdir + im_file + '.png')
    # plt.imsave(outdir + im_file + '.png', img_grad, cmap='gray')

    # outdir_mat = outdir.replace(outdir.split('/')[4], outdir.split('/')[4] + '_matf')
    # if not os.path.exists(outdir_mat):
    #     os.makedirs(outdir_mat)
    #
    # mdic = {"ucm2": img_grad/255.}
    # savemat(outdir_mat + im_file + '.mat', mdic)

    return img_grad


def get_slic_img_contours(im_file, img, regions_slic, graph_raw, perceptual_gradients, gradients_dir, outdir):
    print('##############################', im_file, '##############################')

    if gradients_dir == 'gradients':
        perceptual_gradients = np.sum(perceptual_gradients[:, :-1], axis=-1)

    graph_weighted = graph_raw.copy()

    perceptual_gradients = (perceptual_gradients - min(perceptual_gradients)) / (
                max(perceptual_gradients) - min(perceptual_gradients)) * 255

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_weighted[e[0]][e[1]]['weight'] = perceptual_gradients[i_edge]

    img_grad = graph2gradient(img, graph_weighted, perceptual_gradients, regions_slic)
    img_grad = np.uint8(img_grad)
    ##############################################################################
    '''Visualization Section: show and/or save images'''
    img = Image.fromarray(img_grad)
    img.save(outdir + im_file + '.png')
    # plt.imsave(outdir + im_file + '.png', img_grad, cmap='gray')

    # outdir_mat = outdir.replace(outdir.split('/')[4], outdir.split('/')[4] + '_matf')
    # if not os.path.exists(outdir_mat):
    #     os.makedirs(outdir_mat)
    #
    # mdic = {"ucm2": img_grad/255.}
    # savemat(outdir_mat + im_file + '.mat', mdic)

    return img_grad


def generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, kneighbors=None, n_slic=None,
                                     graph_type=None):
    num_cores = -1

    if kneighbors is not None:
        slic_level = False
        pixel_level = True
        final_dir = str(kneighbors) + 'nn_' + similarity_measure

    elif n_slic is not None and graph_type is not None:
        slic_level = True
        pixel_level = False
        final_dir = str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure
        hdf5_indir_spix = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'superpixels/' +
                               str(n_slic) + '_slic')


    hdf5_indir_im = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_grad = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + gradients_dir + '/' + final_dir)
    hdf5_outdir = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'image_contours/' + final_dir)

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

    if slic_level:
        superpixels_file = h5py.File(hdf5_indir_spix / "Berkeley_superpixels.h5", "r+")
        superpixels_vectors = np.array(superpixels_file["/superpixels"])

        superpixels = np.array(Parallel(n_jobs=num_cores)(
            delayed(np.reshape)(img_spix, (shape[0], shape[1])) for img_spix, shape in
            zip(superpixels_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    if gradients_dir == 'predicted_gradients':
        test_indices = []
        for ii in range(len(images)):
            if img_subdirs[ii] == 'test':
                test_indices.append(ii)

        img_ids = img_ids[test_indices]
        images = images[test_indices]

        ''' Computing Graphs for test set images'''
        if slic_level:
            superpixels = superpixels[test_indices]

            raw_graphs = Parallel(n_jobs=num_cores)(
                delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
                zip(images, superpixels))

        if pixel_level:
            img_shapes = img_shapes[test_indices]
            edges_and_neighbors = Parallel(n_jobs=num_cores)(
                delayed(get_pixel_graph)(kneighbors, img_shape) for img_shape in img_shapes)

        model_input_dirs = sorted(os.listdir(hdf5_indir_grad))
        for mm, model_name in enumerate(model_input_dirs):
            input_directories = sorted(os.listdir(hdf5_indir_grad / model_name))

            for gradients_input_dir in input_directories:
                with h5py.File(hdf5_indir_grad / model_name / gradients_input_dir / 'predicted_gradients.h5',
                               "r+") as gradients_file:
                    print('Reading Berkeley features data set')
                    print('File name: ', gradients_input_dir)

                    imgs_gradients = np.array(gradients_file["/predicted_gradients"])

                    outdir = '../outdir/' + \
                             num_imgs_dir + \
                             'image_contours/' + \
                             final_dir + '/' + \
                             model_name + '/' + \
                             gradients_input_dir + '/'

                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    if slic_level:
                        img_contours = Parallel(n_jobs=num_cores)(
                            delayed(get_slic_img_contours)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                           gradients_dir, outdir)
                            for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                            zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))

                    if pixel_level:
                        img_contours = Parallel(n_jobs=num_cores)(
                            delayed(get_img_contours)(im_file, img_shape, edges_info, perceptual_gradients,
                                                           gradients_dir, outdir)
                            for im_file, img_shape, edges_info, perceptual_gradients in
                            zip(img_ids, img_shapes, edges_and_neighbors, imgs_gradients))

                hdf5_outdir_model = hdf5_outdir / model_name / gradients_input_dir
                hdf5_outdir_model.mkdir(parents=True, exist_ok=True)

                with ImageIndexer(hdf5_outdir_model / 'image_contours.h5',
                                  buffer_size=num_imgs,
                                  num_of_images=num_imgs) as imageindexer:

                    for gradients in img_contours:
                        imageindexer.add(gradients)

    elif gradients_dir == 'gradients':

        ''' Computing Graphs for test set images'''
        if slic_level:

            raw_graphs = Parallel(n_jobs=num_cores)(
                delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
                zip(images, superpixels))

        if pixel_level:
            edges_and_neighbors = Parallel(n_jobs=num_cores)(
                delayed(get_pixel_graph)(kneighbors, img_shape) for img_shape in img_shapes)

        input_directories = sorted(os.listdir(hdf5_indir_grad))

        for gradients_input_dir in input_directories:
            with h5py.File(hdf5_indir_grad / gradients_input_dir / 'gradients.h5', "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_dir)

                gradient_vectors = np.array(gradients_file["/perceptual_gradients"])
                gradient_shapes = np.array(gradients_file["/gradient_shapes"])

                imgs_gradients = Parallel(n_jobs=num_cores)(
                    delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in
                    zip(gradient_vectors, gradient_shapes))

                outdir = '../outdir/' + \
                         num_imgs_dir + \
                         'image_contours/' + \
                         final_dir + '/' + \
                         'SimpleSum_all_imgs/' + \
                         gradients_input_dir + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                if slic_level:
                    img_contours = Parallel(n_jobs=num_cores)(
                        delayed(get_slic_img_contours)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                       gradients_dir, outdir)
                        for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                        zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))

                if pixel_level:
                    img_contours = Parallel(n_jobs=num_cores)(
                        delayed(get_img_contours)(im_file, img_shape, edges_info, perceptual_gradients,
                                                  gradients_dir, outdir)
                        for im_file, img_shape, edges_info, perceptual_gradients in
                        zip(img_ids, img_shapes, edges_and_neighbors, imgs_gradients))


if __name__ == '__main__':
    num_imgs = 7

    kneighbors = 4

    n_slic = 500 * 2

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    gradients_dir = 'predicted_gradients'  # 'predicted_gradients'

    generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, kneighbors=kneighbors, n_slic=None,
                                     graph_type=None)

    generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, kneighbors=None, n_slic=n_slic,
                                     graph_type=graph_type)
