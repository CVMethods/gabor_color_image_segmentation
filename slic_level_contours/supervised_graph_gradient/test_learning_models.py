import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb

from tensorflow.keras.models import Sequential, save_model, load_model

from pathlib import Path
from joblib import Parallel, delayed, dump, load
from sklearn.neighbors import kneighbors_graph

from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale, minmax_scale

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score

sys.path.append('../../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *


class ImageIndexer(object):
    def __init__(self, db_path, buffer_size=200, num_of_images=100):
        self.db = h5py.File(db_path, mode='w')
        self.buffer_size = buffer_size
        self.num_of_images = num_of_images
        self.gradient_arrays_db = None
        self.idxs = {"index": 0}

        self.gradient_arrays_buffer = []

    def __enter__(self):
        print("indexing {} images".format(self.num_of_images))
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gradient_arrays_buffer:
            print("writing last buffers")
            print(len(self.gradient_arrays_buffer))

            self._write_buffer(self.gradient_arrays_db, self.gradient_arrays_buffer)

        print("closing h5 db")
        self.db.close()
        print("indexing took: %.2fs" % (time.time() - self.t0))

    def create_datasets(self):
        self.gradient_arrays_db = self.db.create_dataset(
            "predicted_gradients",
            shape=(self.num_of_images,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=np.dtype('float32'))
        )

    def add(self, feature_array):
        self.gradient_arrays_buffer.append(feature_array.flatten())

        if self.gradient_arrays_db is None:
            self.create_datasets()

        if len(self.gradient_arrays_buffer) >= self.buffer_size:
            self._write_buffer(self.gradient_arrays_db, self.gradient_arrays_buffer)

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


def predicted_slic_gradient_computation(im_file, img, regions_slic, graph_raw, perceptual_gradients, model, sclr, outdir):
    print('##############################', im_file, '##############################')

    X_test = perceptual_gradients[:, :-1]
    graph_pred = graph_raw.copy()

    if isinstance(model, np.ndarray):
        y_pred = np.sum(X_test * model, axis=-1)
    elif not isinstance(model, np.ndarray) and sclr is not None:
        y_pred = model.predict(sclr.fit_transform(X_test))
        y_pred = y_pred.flatten()
    else:
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()

    # y_pred = np.maximum(0, y_pred - np.percentile(y_pred, 5))
    # y_pred = np.minimum(1, 1 * y_pred / np.percentile(y_pred, 95))

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_pred[e[0]][e[1]]['weight'] = y_pred[i_edge]

    save_fig = True
    fontsize = 10
    file_name = im_file

    # Show Graph with updated weights
    fig_title = 'Predicted Gradient'
    img_name = '_weighted_pred'
    colbar_lim = (min(y_pred), max(y_pred))
    show_and_save_imgraph(img, regions_slic, graph_pred, fig_title, img_name, fontsize, save_fig, outdir,
                          file_name, colbar_lim)

    return y_pred


def predicted_gradient_computation(im_file, img_shape, edges_info, perceptual_gradients, model, sclr, outdir):
    print('##############################', im_file, '##############################')

    X_test = perceptual_gradients[:, :-1]

    if isinstance(model, np.ndarray):
        y_pred = np.sum(X_test * model, axis=-1)
    elif not isinstance(model, np.ndarray) and sclr is not None:
        y_pred = model.predict(sclr.fit_transform(X_test))
        y_pred = y_pred.flatten()
    else:
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()

    # y_pred = np.maximum(0, y_pred - np.percentile(y_pred, 5))
    # y_pred = np.minimum(1, 1 * y_pred / np.percentile(y_pred, 95))

    rows, cols, channels = img_shape
    edges_index, neighbors_edges = edges_info

    gradient_pred = np.empty((rows * cols), dtype=np.float32)

    for pp in range(rows * cols):
        gradient_pred[pp] = np.max(y_pred[neighbors_edges[pp]])

    plt.figure(dpi=180)
    plt.imshow(gradient_pred.reshape(rows, cols), cmap='gray')
    plt.savefig(outdir + im_file + '_pred_grad.png')

    # img_grad = (gradient_pred - min(gradient_pred)) / (max(gradient_pred) - min(gradient_pred)) * 255
    # ##############################################################################
    # '''Visualization Section: show and/or save images'''
    # img = Image.fromarray(np.uint8(img_grad.reshape(rows, cols)))
    # img.save(outdir_cont + im_file + '.png')

    return y_pred


def test_models(num_imgs, similarity_measure, kneighbors=None, n_slic=None, graph_type=None):
    num_cores = -1
    source_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

    if kneighbors is not None:
        slic_level = False
        pixel_level = True
        final_dir = str(kneighbors) + 'nn_' + similarity_measure

    elif n_slic is not None and graph_type is not None:
        slic_level = True
        pixel_level = False
        final_dir = str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure
        hdf5_indir_spix = Path(
            source_dir + '../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'superpixels/' + str(
                n_slic) + '_slic')

    hdf5_indir_im = Path(source_dir + '../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_grad = Path(
        source_dir + '../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'gradients/' + final_dir)
    hdf5_outdir = Path(
        source_dir + '../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'predicted_gradients/' + final_dir)
    sav_indir = Path(source_dir+'../../../data/models/' + str(num_imgs) + 'images/' + final_dir)

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

    if slic_level:
        superpixels_file = h5py.File(hdf5_indir_spix / "Berkeley_superpixels.h5", "r+")
        superpixels_vectors = superpixels_file["superpixels"][:]

        superpixels = np.array(Parallel(n_jobs=num_cores)(
            delayed(np.reshape)(img_spix, (shape[0], shape[1])) for img_spix, shape in
            zip(superpixels_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    test_indices = []
    for ii in range(len(images)):
        if img_subdirs[ii] == 'test':
            test_indices.append(ii)

    img_ids = img_ids[test_indices]
    images = images[test_indices]

    ''' Computing Graphs for test set images'''
    if slic_level:
        superpixels = superpixels[test_indices]

        test_raw_graphs = Parallel(n_jobs=num_cores)(
            delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
            zip(images, superpixels))

    if pixel_level:
        img_shapes = img_shapes[test_indices]
        edges_and_neighbors = Parallel(n_jobs=num_cores)(
            delayed(get_pixel_graph)(kneighbors, img_shape) for img_shape in img_shapes)

    input_directories = sorted(os.listdir(hdf5_indir_grad))
    for gradients_input_dir in input_directories:
        with h5py.File(hdf5_indir_grad / gradients_input_dir / 'gradients.h5', "r+") as gradients_file:
            print('Reading Berkeley features data set')
            print('File name: ', gradients_input_dir)
            t0 = time.time()
            gradient_vectors = gradients_file["perceptual_gradients"][:]
            gradient_shapes = gradients_file["gradient_shapes"][:]

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            all_imgs_gradients = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in zip(gradient_vectors, gradient_shapes))

            training_dataset = []
            validation_dataset = []
            all_imgs_gradients_flatten = []
            for ii in range(len(all_imgs_gradients)):
                all_imgs_gradients_flatten.extend(all_imgs_gradients[ii])
                if img_subdirs[ii] == 'train':
                    training_dataset.extend(all_imgs_gradients[ii])
                if img_subdirs[ii] == 'val':
                    validation_dataset.extend(all_imgs_gradients[ii])

            training_dataset = np.array(training_dataset)
            validation_dataset = np.array(validation_dataset)
            all_imgs_gradients_flatten = np.array(all_imgs_gradients_flatten)

            X_train = training_dataset[:, :-1]
            y_train = training_dataset[:, -1]

            X_val = validation_dataset[:, :-1]
            y_val = validation_dataset[:, -1]

            X_train, y_train = balance_classes(X_train, y_train)
            X_val, y_val = balance_classes(X_val, y_val)

            scaler = StandardScaler()  # None  # MinMaxScaler()  #

            if scaler is not None:
                # scaler.fit(all_imgs_gradients_flatten[:, :-1])
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.fit_transform(X_val)

            testing_dataset = np.array(all_imgs_gradients)[test_indices]

            model_input_files = sorted(os.listdir(sav_indir / gradients_input_dir))

            for mm, model_file_name in enumerate(model_input_files):
                model_name = model_file_name.split('.')[0]
                ext = model_file_name.split('.')[1]

                print('Loading model: ' + model_name)
                if ext == 'sav':
                    model = load(sav_indir / gradients_input_dir / model_file_name )
                if ext == 'h5':
                    model = load_model(sav_indir / gradients_input_dir / model_file_name)

                outdir = source_dir+'../../outdir/' + \
                                  num_imgs_dir + \
                                  'predicted_gradients/' + \
                                  final_dir + '/' + \
                                  model_name + '/' + \
                                  gradients_input_dir + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                hdf5_outdir_model = hdf5_outdir / model_name / gradients_input_dir
                hdf5_outdir_model.mkdir(parents=True, exist_ok=True)

                if ext == 'sav':
                    if slic_level:
                        predicted_gradients = Parallel(n_jobs=num_cores)(
                            delayed(predicted_slic_gradient_computation)(im_file, img, regions_slic, graph_raw,
                                                                         perceptual_gradients, model, scaler, outdir)
                            for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                            zip(img_ids, images, superpixels, test_raw_graphs, testing_dataset))
                    if pixel_level:
                        predicted_gradients = Parallel(n_jobs=num_cores)(
                            delayed(predicted_gradient_computation)(im_file, img_shape, edges_info,
                                                                    perceptual_gradients,
                                                                    model, scaler, outdir)
                            for im_file, img_shape, edges_info, perceptual_gradients in
                            zip(img_ids, img_shapes, edges_and_neighbors, testing_dataset))

                elif ext == 'h5':
                    if slic_level:
                        predicted_gradients = Parallel(n_jobs=1)(
                            delayed(predicted_slic_gradient_computation)(im_file, img, regions_slic, graph_raw,
                                                                         perceptual_gradients, model, scaler, outdir)
                            for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                            zip(img_ids, images, superpixels, test_raw_graphs, testing_dataset))
                    if pixel_level:
                        predicted_gradients = Parallel(n_jobs=1)(
                            delayed(predicted_gradient_computation)(im_file, img_shape, edges_info, perceptual_gradients,
                                                                    model, scaler, outdir)
                            for im_file, img_shape, edges_info, perceptual_gradients in
                            zip(img_ids, img_shapes, edges_and_neighbors, testing_dataset))

                with ImageIndexer(hdf5_outdir_model / 'predicted_gradients.h5',
                              buffer_size=num_imgs,
                              num_of_images=num_imgs) as imageindexer:

                    for gradients in predicted_gradients:
                        imageindexer.add(gradients)


if __name__ == '__main__':
    num_imgs = 7

    kneighbors = 4

    n_slic = 500 * 2

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    # test_models(num_imgs, similarity_measure, kneighbors, None, None)
    test_models(num_imgs, similarity_measure, None, n_slic, graph_type)
