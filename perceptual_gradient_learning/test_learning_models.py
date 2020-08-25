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


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')
    sav_dir = Path('../../data/models/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_grad = hdf5_dir / 'complete' / 'gradients'
        sav_indir = sav_dir / 'complete'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_grad = hdf5_dir / '7images' / 'gradients'
        sav_indir = sav_dir / '7images'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_grad = hdf5_dir / '25images' / 'gradients'
        sav_indir = sav_dir / '25images'
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

    graph_type = 'knn'
    kneighbors = 8
    radius = 10

    input_files = os.listdir(hdf5_indir_grad)
    for gradients_input_file in input_files:
        with h5py.File(hdf5_indir_grad / gradients_input_file, "r+") as gradients_file:
            print('Reading Berkeley features data set')
            print('File name: ', gradients_input_file)
            t0 = time.time()
            gradient_vectors = np.array(gradients_file["/perceptual_gradients"])
            gradient_shapes = np.array(gradients_file["/gradient_shapes"])
            superpixel_vectors = np.array(gradients_file["/superpixels"])
            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            all_imgs_data = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in zip(gradient_vectors, gradient_shapes))

            input_file_name = gradients_input_file.split('_')
            input_file_name[1] = 'Models'
            input_model_dir = '_'.join(input_file_name)[:-3]
            model_input_files = sorted(os.listdir(sav_indir / input_model_dir))

            output_file_name = gradients_input_file.split('_')
            output_file_name[1] = 'PredictedGradients'
            output_file = '_'.join(output_file_name)[:-3]

            for mm, model_file_name in enumerate(model_input_files):

                outdir = outdir = '../outdir/perceptual_gradient/' + \
                                  num_imgs_dir + \
                                  'model_predictions/' + \
                                  output_file + '/' + \
                                  model_file_name[:-4] + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                print('Loading model: ' + model_file_name[:-4])
                model = load(sav_indir / input_model_dir / model_file_name)

                testing_dataset = []
                X_test = []
                y_test = []
                for ii in range(len(all_imgs_data)):
                    if img_subdirs[ii] == 'dir':  # Need to change the name of directory to add the gradients to training dataset
                        # testing_dataset.append(all_imgs_data[ii])
                        X_test = all_imgs_data[ii][:, :-1]
                        y_test = all_imgs_data[ii][:, -1]

                        pred = model.predict(X_test)
                        normalized_pred = (pred-min(pred))/(max(pred)-min(pred))

                        mae_score = 1. - mean_absolute_error(y_test, normalized_pred)
                        r2score = r2_score(y_test, normalized_pred)
                        print('Score :',  1. - mae_score, r2score)

                        regions_slic = superpixel_vectors[ii].reshape((img_shapes[ii][0], img_shapes[ii][1]))
                        graph_pred = get_graph(images[ii], regions_slic, graph_type, kneighbors, radius)
                        # graph_pred = graph_raw.copy()

                        for i_edge, e in enumerate(list(graph_pred.edges)):
                            graph_pred[e[0]][e[1]]['weight'] = normalized_pred[i_edge]

                         # Segmentation parameters
                        method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence
                        graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
                        law_type = 'gamma'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
                        cut_level = 0.9  # set threshold at the 90% quantile level

                        graph_mst = get_mst(graph_pred)
                        weights = nx.get_edge_attributes(graph_mst, 'weight').values()
                        graph_weighted = graph_mst
                        thresh, params = fit_distribution_law(list(weights), cut_level, law_type)
                        graph_aftercut = graph_weighted.copy()
                        graph.cut_threshold(regions_slic, graph_aftercut, thresh, in_place=True)
                        regions_aftercut = graph2regions(graph_aftercut, regions_slic)

                        pdb.set_trace()
                        # Visualization Params
                        save_fig = True
                        fontsize = 10
                        file_name = img_ids[ii]

                        # Show Graph with updated weights
                        fig_title = model_file_name[:-4] +' Predicted Gradient (' + graph_type + ')'
                        img_name = '_weighted_pred_' + graph_type
                        colbar_lim = (min(pred), max(pred))
                        show_and_save_imgraph(images[ii], regions_slic, graph_pred, fig_title, img_name, fontsize, save_fig, outdir,
                              file_name, colbar_lim)

                        plt.clf()
                        plt.cla()
                        plt.close('all')