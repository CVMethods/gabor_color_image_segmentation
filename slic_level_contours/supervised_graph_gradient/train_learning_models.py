import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb
import seaborn as sns

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sknn.mlp import Regressor, Layer
sys.path.append('../../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *


def train_models(num_imgs, n_slic, graph_type, similarity_measure):
    num_cores = -1
    source_dir = os.path.dirname(os.path.abspath(__file__))+'/'
    hdf5_indir_im = Path(source_dir+'../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_grad = Path(source_dir+'../../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'gradients/' +
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

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

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
            for ii in range(len(all_imgs_gradients)):
                if img_subdirs[ii] == 'train':  # Need to change the name of directory to add the gradients to training dataset
                    training_dataset.extend(all_imgs_gradients[ii])

            training_dataset = np.array(training_dataset)
            X_train = training_dataset[:, :-1]
            y_train = training_dataset[:, -1]

            X_train, y_train = balance_classes(X_train, y_train)


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
            # sns.distplot(y_train_balanced)
            # plt.figure()
            # plt.scatter(X_train[:, 0], y_train, c='red', marker='.', cmap=plt.cm.flag, label='imbalanced data')
            # plt.figure()
            # plt.scatter(X_train[:, 0], y_train_balanced, c='blue', marker='.', cmap=plt.cm.flag, label='imbalanced data')
            #
            #
            # plt.figure()
            # plt.scatter(X_bal[:, 0], y_bal, c='green', marker='.', cmap=plt.cm.flag, label='imbalanced data')
            # plt.show(block=False)
            #
            # pdb.set_trace()

            validation_dataset = []
            for ii in range(len(all_imgs_gradients)):
                if img_subdirs[ii] == 'val':  # Need to change the name of directory to add the gradients to training dataset
                    validation_dataset.extend(all_imgs_gradients[ii])

            validation_dataset = np.array(training_dataset)
            X_val = training_dataset[:, :-1]
            y_val = training_dataset[:, -1]

            X_val, y_val = balance_classes(X_val, y_val)

            #y_val_balanced = compute_sample_weight('balanced', y_train)
            # class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)

            regressors = [('LinReg', LinearRegression(n_jobs=-1)),
                          # ('Elastic', ElasticNet(random_state=5)),
                          # ('Ridge', Ridge(alpha=5)),
                          ('SGDR', SGDRegressor(loss='epsilon_insensitive', penalty='elasticnet', alpha=0.0001, verbose=1, random_state=1, learning_rate='invscaling', early_stopping=True, validation_fraction=0.3, n_iter_no_change=5)),
                          # ('SVR', LinearSVR(epsilon=0.0, loss='epsilon_insensitive', verbose=1, random_state=1)),
                          ('MLPR', Regressor(layers=[Layer(type="Rectifier", units=16), Layer(type="Rectifier", units=8), Layer(type="Linear", units=1)],
                                                      random_state=1,
                                                      learning_rule='sgd',
                                                      learning_rate=0.0001,
                                                      batch_size=1400,
                                                      valid_set=(X_val, y_val),
                                                      loss_type='mse',
                                                      verbose=True))]

            outdir = source_dir+'../../../data/models/' + \
                     num_imgs_dir + \
                     (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                     gradients_input_dir + '/'

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            for name, regressor in regressors:
                print('Performing ' + name)
                reg = Pipeline([('scl', MinMaxScaler()), (name, regressor)])
                t0 = time.time()
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
                dump(reg, outdir + filename)


if __name__ == '__main__':

    num_imgs = 7
    n_slic = 500 * 4

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'eps'

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    train_models(num_imgs, n_slic, graph_type, similarity_measure)