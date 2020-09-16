import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb

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
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score

sys.path.append('../')
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


def predicted_gradient_computation(im_file, img, regions_slic, graph_raw, perceptual_gradients, model, outdir):
    print('##############################', im_file, '##############################')

    graph_pred = graph_raw.copy()
    X_test = perceptual_gradients[:, :-1]
    y_test = perceptual_gradients[:, -1]
    #
    y_pred = model.predict(X_test)

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_pred[e[0]][e[1]]['weight'] = y_pred[i_edge]

    img_grad = graph2gradient(img, graph_pred, y_pred, regions_slic)

    save_fig = True
    fontsize = 10
    file_name = im_file

    # Show Graph with updated weights
    fig_title = model_file_name[:-4] + ' Predicted Gradient (' + graph_type + ')'
    img_name = '_weighted_pred_' + graph_type
    colbar_lim = (min(y_pred), max(y_pred))
    show_and_save_imgraph(img, regions_slic, graph_pred, fig_title, img_name, fontsize, save_fig, outdir,
                          file_name, colbar_lim)

    plt.figure(dpi=180)
    plt.imshow(img_grad, cmap='gray')
    plt.savefig(outdir + im_file + '_grad_pred_' + graph_type + '.png')

    return y_pred


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')
    sav_dir = Path('../../data/models/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_spix = hdf5_dir / 'complete' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / 'complete' / 'gradients'
        sav_indir = sav_dir / 'complete'
        hdf5_outdir = hdf5_dir / 'complete' / 'predicted_gradients'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_spix = hdf5_dir / '7images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '7images' / 'gradients'
        sav_indir = sav_dir / '7images'
        hdf5_outdir = hdf5_dir / '7images' / 'predicted_gradients'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_spix = hdf5_dir / '25images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '25images' / 'gradients'
        sav_indir = sav_dir / '25images'
        hdf5_outdir = hdf5_dir / '25images' / 'predicted_gradients'
        num_imgs_dir = '25images/'

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
    n_slic_regions = str(superpixels_file.attrs['num_slic_regions']) + '_regions'

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    test_indices = []
    for ii in range(len(images)):
        if img_subdirs[ii] == 'test':  # Need to change the name of directory to add the gradients to training dataset
            test_indices.append(ii)

    img_ids = img_ids[test_indices]
    images = images[test_indices]
    superpixels = superpixels[test_indices]

    ''' Computing Graphs for test set images'''
    # Graph function parameters
    graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
    kneighbors = 8
    radius = 10

    test_raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type, kneighbors, radius) for img, regions_slic in
        zip(images, superpixels))

    input_files = os.listdir(hdf5_indir_grad)
    for gradients_input_file in input_files:
        with h5py.File(hdf5_indir_grad / gradients_input_file, "r+") as gradients_file:
            print('Reading Berkeley features data set')
            print('File name: ', gradients_input_file)
            t0 = time.time()
            gradient_vectors = np.array(gradients_file["/perceptual_gradients"])
            gradient_shapes = np.array(gradients_file["/gradient_shapes"])

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            all_imgs_gradients = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in zip(gradient_vectors, gradient_shapes))

            testing_dataset = np.array(all_imgs_gradients)[test_indices]

            input_file_name = gradients_input_file.split('_')
            input_file_name[1] = 'Models'
            input_model_dir = '_'.join(input_file_name)[:-3]
            model_input_files = sorted(os.listdir(sav_indir / input_model_dir / n_slic_regions))

            output_file_name = gradients_input_file.split('_')
            output_file_name[1] = 'PredictedGradients'
            output_file = '_'.join(output_file_name)

            for mm, model_file_name in enumerate(model_input_files):
                print('Loading model: ' + model_file_name[:-4])
                model = load(sav_indir / input_model_dir / n_slic_regions / model_file_name)

                outdir = outdir = '../outdir/perceptual_gradient/' + \
                                  num_imgs_dir + \
                                  'model_predictions/' + \
                                  output_file[:-3] + '/' + \
                                  n_slic_regions + '/' + \
                                  model_file_name[:-4] + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                hdf5_outdir_model = hdf5_outdir / model_file_name[:-4]
                hdf5_outdir_model.mkdir(parents=True, exist_ok=True)

                predicted_gradients = Parallel(n_jobs=num_cores)(
                    delayed(predicted_gradient_computation)(im_file, img, regions_slic, graph_raw, perceptual_gradients, model, outdir)
                    for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                    zip(img_ids, images, superpixels, test_raw_graphs, testing_dataset))

                with ImageIndexer(hdf5_outdir_model / output_file,
                              buffer_size=num_imgs,
                              num_of_images=num_imgs) as imageindexer:

                    for gradients in predicted_gradients:
                        imageindexer.add(gradients)
                        imageindexer.db.attrs['num_slic_regions'] = n_regions