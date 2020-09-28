import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


from pathlib import Path
from joblib import Parallel, delayed, dump
from sklearn.neighbors import kneighbors_graph

from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet, LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale, minmax_scale
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sknn.mlp import Regressor, Layer

sys.path.append('../../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *
from source.plot_save_figures import *


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


def predicted_gradient_computation(im_file, img, regions_slic, graph_raw, perceptual_gradients, model, sclr, outdir):
    print('##############################', im_file, '##############################')

    graph_pred = graph_raw.copy()
    X_test = sclr.transform(perceptual_gradients[:, :-1])
    # y_test = perceptual_gradients[:, -1]
    #
    if hasattr(model, 'steps'):
        y_pred = model.predict(X_test)
    # y_pred = (y_pred - min(y_pred)) / (max(y_pred) - min(y_pred))
        if model.steps[0][0] == 'MLPR':
            y_pred = y_pred.flatten() #1 -
    else:
        # y_pred = model.predict(minmax_scale(X_test))
        y_pred = model.predict(X_test)
        y_pred = y_pred.flatten()

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_pred[e[0]][e[1]]['weight'] = y_pred[i_edge]

    # img_grad = graph2gradient(img, graph_pred, y_pred, regions_slic)

    save_fig = True
    fontsize = 10
    file_name = im_file

    # Show Graph with updated weights
    fig_title = 'Predicted Gradient'
    img_name = '_weighted_pred'
    colbar_lim = (min(y_pred), max(y_pred))
    show_and_save_imgraph(img, regions_slic, graph_pred, fig_title, img_name, fontsize, save_fig, outdir,
                          file_name, colbar_lim)

    # plt.figure(dpi=180)
    # plt.imshow(img_grad, cmap='gray')
    # plt.savefig(outdir + im_file + '_grad_pred_' + graph_type + '.png')

    return y_pred


def BuildModel(X):
    kernel_init = 'normal'

    # it is a sequential model
    model = Sequential()
    model.add(Dense(128, input_dim=len(X[0, :]), activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(64, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(32, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(16, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(8, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(4, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(1, activation='linear', kernel_initializer=kernel_init))
    model.compile(loss=MeanSquaredError(), optimizer='adam')
    return model


def SmallModel(X):
    kernel_init = 'normal'

    # it is a sequential model
    model = Sequential()
    model.add(Dense(8, input_dim=len(X[0, :]), activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(4, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(1, activation='relu', kernel_initializer=kernel_init))
    model.compile(loss=MeanSquaredError(), optimizer='adam')
    return model


def train_test_models(num_imgs, n_slic, graph_type, similarity_measure):
    num_cores = -1
    source_dir = os.path.dirname(os.path.abspath(__file__))+'/'
    hdf5_indir_im = Path(source_dir+'../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_spix = Path(source_dir+'../../../data/hdf5_datasets/'+str(num_imgs)+'images/' + 'superpixels/'+str(n_slic)+'_slic')
    hdf5_indir_grad = Path(source_dir+'../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'gradients/' +
                           str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure)
    hdf5_outdir = Path(source_dir+'../../../data/hdf5_datasets/'+str(num_imgs)+'images/' + 'predicted_gradients/' +
                       str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure)

    num_imgs_dir = str(num_imgs) + 'images/'

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
    test_raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
        zip(images, superpixels))

    input_directories = sorted(os.listdir(hdf5_indir_grad))
    for gradients_input_dir in input_directories:
        with h5py.File(hdf5_indir_grad / gradients_input_dir / 'gradients.h5', "r+") as gradients_file:
            print('Reading Berkeley features data set')
            print('File name: ', gradients_input_dir)
            t0 = time.time()
            gradient_vectors = np.array(gradients_file["/perceptual_gradients"])
            gradient_shapes = np.array(gradients_file["/gradient_shapes"])
            n_regions = gradients_file.attrs['num_slic_regions']

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            all_imgs_gradients = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in zip(gradient_vectors, gradient_shapes))

            training_dataset = []
            validation_dataset = []
            for ii in range(len(all_imgs_gradients)):
                if img_subdirs[ii] == 'train':
                    training_dataset.extend(all_imgs_gradients[ii])
                if img_subdirs[ii] == 'val':
                    validation_dataset.extend(all_imgs_gradients[ii])

            training_dataset = np.array(training_dataset)
            validation_dataset = np.array(validation_dataset)

            # if graph_type.endswith('nn'):
            #     neighbors = int("".join(list(graph_type)[:-2]))
            #     num_edges = (2 * neighbors) - 1
            #     X_train = training_dataset[:, :num_edges*3]
            #     y_train = training_dataset[:, num_edges*3:]
            #     X_val = validation_dataset[:, :num_edges*3]
            #     y_val = validation_dataset[:, num_edges*3:]
            #
            # else:
            X_train = training_dataset[:, :-1]
            y_train = training_dataset[:, -1]

            X_val = validation_dataset[:, :-1]
            y_val = validation_dataset[:, -1]

            X_train, y_train = balance_classes(X_train, y_train)
            X_val, y_val = balance_classes(X_val, y_val)

            batch_sz = int(len(X_train)/(num_imgs))
            # y_train_balanced = compute_sample_weight('balanced', y_train)
            # class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
            # plt.figure()
            # sns.distplot(X_train[:, 0])
            # plt.figure()
            # sns.distplot(X_train[:, 1])
            # plt.figure()
            # sns.distplot(X_train[:, 2])
            # plt.figure()
            # sns.distplot(y_train)
            # plt.figure()
            # plt.scatter(X_train[:, 0], y_train, c='red', marker='.', cmap=plt.cm.flag, label='imbalanced data')
            # plt.show(block=False)
            # pdb.set_trace()
            #
            # y_train_balanced = compute_sample_weight('balanced', y_train)
            # plt.figure()
            # plt.scatter(X_train[:, 0], y_train_balanced, c='red', marker='.', cmap=plt.cm.flag, label='imbalanced weigthed data')
            # plt.figure()
            # sns.distplot(y_train_balanced)
            # plt.show(block=False)
            # pdb.set_trace()
            #
            # X_train, y_train = balance_classes(X_train, y_train)
            #
            # plt.figure()
            # sns.distplot(X_train[:, 0])
            # plt.figure()
            # sns.distplot(X_train[:, 1])
            # plt.figure()
            # sns.distplot(X_train[:, 2])
            # plt.figure()
            # sns.distplot(y_train)
            # plt.show(block=False)
            # pdb.set_trace()
            #
            # plt.figure()
            # plt.scatter(X_train[:, 0], y_train, c='red', marker='.', cmap=plt.cm.flag, label='balanced data')
            # plt.show(block=False)
            # pdb.set_trace()

            scaler = StandardScaler()  # MinMaxScaler()  #
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

            testing_dataset = np.array(all_imgs_gradients)[test_indices]

            regressors = [
                          ('LinReg', LinearRegression(n_jobs=num_cores)),
                          # ('Elastic', ElasticNet(random_state=5)),
                          # ('Ridge', Ridge(alpha=5)),
                          ('SGDR', SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.0001, verbose=1, random_state=1, learning_rate='invscaling', early_stopping=True, validation_fraction=0.3, n_iter_no_change=10)),
                          # ('SVR', LinearSVR(epsilon=0.0, loss='epsilon_insensitive', verbose=1, random_state=1)),
                          # ('MLPR', Regressor(layers=[Layer(type="Rectifier", units=8),
                          #                            Layer(type="Rectifier", units=4),
                          #                            Layer(type="Linear", units=1)],
                          #                             random_state=1,
                          #                             learning_rule='sgd',
                          #                             learning_rate=0.01,
                          #                             batch_size=batch_sz,
                          #                             n_iter=100,
                          #                             n_stable=4,
                          #                             valid_set=(X_val, y_val), #minmax_scale()
                          #                             loss_type='mse',
                          #                             verbose=True)),
                          ('MLPR_tf', SmallModel(X_train))
                          ]

            outdir_models = source_dir+'../../../data/models/' + \
                     num_imgs_dir + \
                     (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                     gradients_input_dir + '/'

            if not os.path.exists(outdir_models):
                os.makedirs(outdir_models)

            for name, regressor in regressors:

                outdir = source_dir+'../../outdir/' + \
                                  num_imgs_dir + \
                                  'predicted_gradients/' + \
                                  (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                                  name + '/' + \
                                  gradients_input_dir + '/'
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                hdf5_outdir_model = hdf5_outdir / name / gradients_input_dir
                hdf5_outdir_model.mkdir(parents=True, exist_ok=True)

                print('Performing ' + name)
                reg = Pipeline([(name, regressor)])#('scl', MinMaxScaler()),
                t0 = time.time()
                if name == 'MLPR_tf':
                    print(regressor.summary())

                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=4, min_delta=1E-4, restore_best_weights=True)
                    mc = ModelCheckpoint(outdir_models + name + '.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
                    # fit_params = {name + '__callbacks': [es, mc]}
                    # reg.fit(X_train, y_train, **fit_params)
                    history = regressor.fit(x=X_train, y=y_train, batch_size=batch_sz, epochs=300, verbose=1, callbacks=[es, mc],
                             validation_data=(X_val, y_val), shuffle=True, sample_weight=None)

                    # summarize history for loss
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend(['train', 'validation'], loc='upper left')
                    plt.savefig(outdir + 'mlp_model_loss.png')

                    predicted_gradients = Parallel(n_jobs=1)(
                        delayed(predicted_gradient_computation)(im_file, img, regions_slic, graph_raw,
                                                                perceptual_gradients, regressor, scaler, outdir)
                        for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                        zip(img_ids, images, superpixels, test_raw_graphs, testing_dataset))

                else:
                    if name == 'MLPR':
                        #fit_params = {name + '__w': y_train_balanced}
                        # reg.fit(X_train, y_train, **fit_params)
                        reg.fit(X_train, y_train)
                    else:
                        #fit_params = {name + '__sample_weight': y_train_balanced}
                        # reg.fit(X_train, y_train, **fit_params)
                        X_train = np.concatenate((X_train, X_val))
                        y_train = np.concatenate((y_train, y_val))
                        reg.fit(X_train, y_train)
                        print('Coefficients', name, ' :', reg[name].coef_)

                    train_time = time.time() - t0
                    print("train time: %0.3fs" % train_time)
                    filename = name + '.sav'
                    dump(reg, outdir_models + filename)

                    predicted_gradients = Parallel(n_jobs=num_cores)(
                        delayed(predicted_gradient_computation)(im_file, img, regions_slic, graph_raw,
                                                                perceptual_gradients, reg, scaler, outdir)
                        for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                        zip(img_ids, images, superpixels, test_raw_graphs, testing_dataset))

                with ImageIndexer(hdf5_outdir_model / 'predicted_gradients.h5',
                                  buffer_size=num_imgs,
                                  num_of_images=num_imgs) as imageindexer:

                    for gradients in predicted_gradients:
                        imageindexer.add(gradients)
                        imageindexer.db.attrs['num_slic_regions'] = n_regions


if __name__ == '__main__':

    num_imgs = 7
    n_slic = 100 * 2

    # Graph function parameters
    graph_type = '4nn'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    train_test_models(num_imgs, n_slic, graph_type, similarity_measure)