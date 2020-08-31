import os
import time
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pdb

from pathlib import Path
from joblib import Parallel, delayed, dump
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

sys.path.append('../')
from source.groundtruth import *
from source.computation_support import *
from source.graph_operations import *


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_grad = hdf5_dir / 'complete' / 'gradients'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_grad = hdf5_dir / '7images' / 'gradients'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_grad = hdf5_dir / '25images' / 'gradients'
        num_imgs_dir = '25images/'

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

    input_files = os.listdir(hdf5_indir_grad)
    for gradients_input_file in input_files:
        with h5py.File(hdf5_indir_grad / gradients_input_file, "r+") as gradients_file:
            print('Reading Berkeley features data set')
            print('File name: ', gradients_input_file)
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

            regressors = [('SGDR', SGDRegressor(loss='huber', max_iter=5000, tol=1e-3)),
                          ('MLPR', MLPRegressor(solver='adam', activation='relu', max_iter=5000)),
                          ('LinReg', LinearRegression(n_jobs=-1)),
                          ('Ridge', Ridge(alpha=2.0))]

            output_file_name = gradients_input_file.split('_')
            output_file_name[1] = 'Models'
            output_file = '_'.join(output_file_name)

            outdir = '../../data/models/' + num_imgs_dir + output_file[:-3] + '/%d_regions/' % n_regions

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            for name, regressor in regressors:
                print('Performing ' + name)
                reg = regressor
                reg = make_pipeline(StandardScaler(), reg)
                t0 = time.time()
                reg.fit(X_train, y_train)
                train_time = time.time() - t0
                print("train time: %0.3fs" % train_time)

                filename = name + '.sav'
                dump(reg, outdir + filename)