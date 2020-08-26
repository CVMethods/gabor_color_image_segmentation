"""
=========================================================
Comparing different clustering algorithms on BSD images
=========================================================

Ce script est inspiré de comparaison de différentes techniques de clustering en scikit-learn et adapté aux images. Le clustering, très basique, se fait sur la base des charactéritiques de Gabor.

Il n'y a que deux algorithmes assez rapide pour marcher sur les pixels. Cependant, d'autres clustering techniques peuvent être testés après sous-échantillonnage de l'image d'entrée. Il faudra tester ces algorithmes sur l'aggrégation de superpixels.

Pour certain, il faut spécifier le nombre de clusters.
"""
print(__doc__)

import sys
import h5py
import time
import warnings
import os
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from joblib import Parallel, delayed
from scipy.cluster import vq
from scipy.stats import hmean
from sklearn.decomposition import PCA
from skimage import segmentation, color, data
from skimage.io import imread
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from petastorm import make_reader
from PIL import Image

sys.path.append('../')
from source.groundtruth import *
from source.metrics import *
from source.computation_support import *

def clustering_segmentation_and_metrics(i_dataset, dataset, algo_params, num_clusters):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    img_id, img, X, y, n_clusters, img_size = dataset

    print('dataset image:', img_id)

    # img_size = img_size[:-1]
    rows, cols, channels = img_size

    if num_clusters == 'max':
        params['n_clusters'] = int(n_clusters[0])
    elif num_clusters == 'min':
        params['n_clusters'] = int(n_clusters[1])
    elif num_clusters == 'mean':
        params['n_clusters'] = int(n_clusters[2])
    elif num_clusters == 'hmean':
        params['n_clusters'] = int(n_clusters[3])

    # print(params['n_clusters'], num_clusters, int(n_clusters[0]))

    # Add pixel's position to features to include locality
    pixels = np.arange(rows * cols)
    nodes = pixels.reshape((rows, cols))
    yy, xx = np.where(nodes >= 0)
    X = np.column_stack((X, yy, xx))

    # # normalize dataset for easier parameter selection
    # X = StandardScaler().fit_transform(X)
    X = vq.whiten(X)

    # Reduce data dimensionality (if needed for faster clustering computation)
    pca = PCA(n_components=4)
    X = pca.fit_transform(X)

    # # estimate bandwidth for mean shift
    # bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # # connectivity matrix for structured Ward
    # connectivity = kneighbors_graph(
    #     X, n_neighbors=params['n_neighbors'], include_self=False)
    # # make connectivity symmetric
    # connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'], random_state=0)
    k_means = cluster.KMeans(params['n_clusters'], random_state=0, n_jobs=params['n_jobs'])
    # ward = cluster.AgglomerativeClustering(
    #     n_clusters=params['n_clusters'], linkage='ward',
    #     connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'],
                            xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'],
                            n_jobs=params['n_jobs'])
    # affinity_propagation = cluster.AffinityPropagation(
    #     damping=params['damping'], preference=params['preference'])
    # average_linkage = cluster.AgglomerativeClustering(
    #     linkage="average", affinity="cityblock",
    #     n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    # uncomment to try other clustering algorithms
    clustering_algorithms = (
        ('KMeans', k_means),
        ('MiniBatchKMeans', two_means),
        # ('AffinityPropagation', affinity_propagation), # commented out since excessively long
        # ('MeanShift', ms), # commented out since too long bandwidth computation
        # ('SpectralClustering', spectral), # commented cause difficult to have it working
        # ('Ward', ward), # commented out since too long connectivity computation
        # ('AgglomerativeClustering', average_linkage), # commented out since too long connectivity computation
        # ('DBSCAN', dbscan), # commented out since memory consuming
        # ('OPTICS', optics), # commented out since too long (even with parallel jobs)
        ('Birch', birch),
        ('GaussianMixture', gmm)  # commented out since too long on big images
    )
    segmentations = []
    img_metrics = []
    for algo_name, algorithm in clustering_algorithms:
        t0 = time.time()
        print('algorithm %s' % algo_name)

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                category=UserWarning)

            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        nc = len(np.unique(y_pred))
        y_pred = y_pred.reshape((rows, cols))

        # Evaluate metrics
        m = metrics(None, y_pred, y)
        m.set_metrics()
        # m.display_metrics()

        metrics_values = m.get_metrics()

        plt.figure(dpi=180)
        out = color.label2rgb(y_pred, img, kind='avg')
        out = segmentation.mark_boundaries(out, y_pred, color=(0, 0, 0), mode='thick')
        ax = plt.gca()
        im = ax.imshow(out)#, cmap=plt.cm.get_cmap('jet', nc)
        ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                       length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
        ax.set_title(algo_name + ' k=%d' % nc, fontsize=10)
        ax.set_xlabel(('Recall: %.3f, Precision: %.3f, Time: %.2fs' % (
            metrics_values['recall'], metrics_values['precision'], (t1 - t0))).lstrip('0'), fontsize=10)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cb = plt.colorbar(im, cax=cax, ticks=(np.arange(nc) + 0.5) * (nc - 1) / nc)
        # cb.set_label(label='$labels$', fontsize=10)
        # cb.ax.tick_params(labelsize=6)
        # cb.ax.set_yticklabels([r'${{{}}}$'.format(val) for val in range(nc)])
        plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_' + algo_name + '_' + num_clusters + '_segm.png')
        plt.cla()
        plt.clf()
        plt.close()

        img_metrics.append(metrics_values)
        segmentations.append(y_pred)

    # return segmentations, img_metrics
    return img_metrics



def prepare_dataset(img_id, image, gabor_features, img_shape):
    ground_truth = np.array(get_segment_from_filename(img_id))
    n_segments = get_num_segments(ground_truth)
    return (img_id, image, gabor_features, ground_truth, n_segments, img_shape), {}


if __name__ == '__main__':
    np.random.seed(0)
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_feat = hdf5_dir / 'complete' / 'features'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_feat = hdf5_dir / '7images/' / 'features'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images/' / 'images'
        hdf5_indir_feat = hdf5_dir / '25images/' / 'features'
        num_imgs_dir = '25images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = np.array(images_file["/images"])
    img_shapes = np.array(images_file["/image_shapes"])
    img_ids = np.array(images_file["/image_ids"])

    images = Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    input_files = os.listdir(hdf5_indir_feat)
    for features_input_file in input_files:
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = np.array(features_file["/gabor_features"])
            feature_shapes = np.array(features_file["/feature_shapes"])

            features = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1])) for features, shape in zip(feature_vectors, feature_shapes))
            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            iterator = zip(img_ids, images, features, img_shapes)

            datasets = Parallel(n_jobs=num_cores)(
                delayed(prepare_dataset)(im_id, image, feature, shape) for im_id, image, feature, shape in iterator)

            default_base = {'quantile': .3,
                            'eps': .3,
                            'damping': .9,
                            'preference': -200,
                            'n_neighbors': 10,
                            'n_clusters': 4,
                            'min_samples': 20,
                            'xi': 0.05,
                            'min_cluster_size': 0.1,
                            'n_jobs': 10}

            possible_num_clusters = ['max', 'min', 'mean', 'hmean', 'const']
            for num_clusters in possible_num_clusters:

                outdir = '../outdir/pixel_level_segmentation/' + num_imgs_dir + 'all_methods_benchmark/' + features_input_file[:-3] + '/' + num_clusters + '_nclusters/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                segmentation_metrics = Parallel(n_jobs=num_cores, prefer='processes')(
                    delayed(clustering_segmentation_and_metrics)(i_dataset + 1, dataset, algo_params, num_clusters) for
                    i_dataset, (dataset, algo_params) in enumerate(datasets))

                KMeans_metrics = []
                MiniBatchKMeans_metrics = []
                Birch_metrics = []
                GaussianMixture_metrics = []
                all_precisions = []
                all_recalls = []

                for ii in range(len(datasets)):
                    algo_metrics = segmentation_metrics[ii]
                    KMeans_metrics.append((algo_metrics[0]['recall'], algo_metrics[0]['precision'], algo_metrics[0]['underseg'], algo_metrics[0]['undersegNP'], algo_metrics[0]['compactness'], algo_metrics[0]['density']))
                    MiniBatchKMeans_metrics.append((algo_metrics[1]['recall'], algo_metrics[1]['precision'], algo_metrics[1]['underseg'], algo_metrics[1]['undersegNP'], algo_metrics[1]['compactness'], algo_metrics[1]['density']))
                    Birch_metrics.append((algo_metrics[2]['recall'], algo_metrics[2]['precision'], algo_metrics[2]['underseg'], algo_metrics[2]['undersegNP'], algo_metrics[2]['compactness'], algo_metrics[2]['density']))
                    GaussianMixture_metrics.append((algo_metrics[3]['recall'], algo_metrics[3]['precision'], algo_metrics[3]['underseg'], algo_metrics[3]['undersegNP'], algo_metrics[3]['compactness'], algo_metrics[3]['density']))

                    all_precisions.extend([algo_metrics[0]['precision'], algo_metrics[1]['precision'], algo_metrics[2]['precision'], algo_metrics[3]['precision']])
                    all_recalls.extend([algo_metrics[0]['recall'], algo_metrics[1]['recall'], algo_metrics[2]['recall'], algo_metrics[3]['recall']])

                KMeans_metrics = np.array(KMeans_metrics)
                MiniBatchKMeans_metrics = np.array(MiniBatchKMeans_metrics)
                Birch_metrics = np.array(Birch_metrics)
                GaussianMixture_metrics = np.array(GaussianMixture_metrics)

                np.savetxt(outdir + 'KMeans_metrics.csv', np.column_stack((img_ids, KMeans_metrics)), delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f'], header='img ID, recall, precision, undersegmentation Bergh, undersegmentation NP, compactness, density', comments='')
                np.savetxt(outdir + 'MiniBatchKMeans_metrics.csv', np.column_stack((img_ids, MiniBatchKMeans_metrics)), delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f'], header='img ID, recall, precision, undersegmentation Bergh, undersegmentation NP, compactness, density', comments='')
                np.savetxt(outdir + 'Birch_metrics.csv', np.column_stack((img_ids, Birch_metrics)), delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f'], header='img ID, recall, precision, undersegmentation Bergh, undersegmentation NP, compactness, density', comments='')
                np.savetxt(outdir + 'GaussianMixture_metrics.csv', np.column_stack((img_ids, GaussianMixture_metrics)), delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f'], header='img ID, recall, precision, undersegmentation Bergh, undersegmentation NP, compactness, density', comments='')

                algorithms_metrics = [KMeans_metrics, MiniBatchKMeans_metrics, Birch_metrics, GaussianMixture_metrics]
                algorithms_names = ['KMeans', 'MiniBatchKMeans', 'Birch', 'GaussianMixture']

                score_method_names = algorithms_names * len(datasets)
                df = pd.DataFrame({'Method': score_method_names, 'Precision': all_precisions, 'Recall': all_recalls})
                df = df[['Method', 'Precision', 'Recall']]
                dd = pd.melt(df, id_vars=['Method'], value_vars=['Precision', 'Recall'], var_name='scores')

                plt.figure(dpi=180)
                sns.boxplot(x='Method', y='value', data=dd, hue='scores')
                plt.title('Precision / Recall box plot ' + num_clusters + ' nclusters')
                plt.savefig(outdir + 'PrecisionRecall_boxplot_' + num_clusters + '_nclusters.png', bbox_inches='tight')

                for name, result_metrics in zip(algorithms_names, algorithms_metrics):
                    recall = result_metrics[:, 0]
                    precision = result_metrics[:, 1]

                    plt.figure(dpi=180)
                    plt.plot(np.arange(len(datasets)) + 1, recall, '-o', c='k', label='recall')
                    plt.plot(np.arange(len(datasets)) + 1, precision, '-o', c='r', label='precision')
                    plt.title(name + ' P/R histogram ' + num_clusters + ' nclusters')
                    plt.xlabel(
                        'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, '
                        'Pmed: %.3f, Pstd: %.3f ' % (recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(),
                                                     precision.max(), precision.min(), precision.mean(), np.median(precision),
                                                     precision.std()))
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.grid()
                    plt.savefig(outdir + name + '_PR_hist_' + num_clusters + '_nclusters.png', bbox_inches='tight')
                    plt.cla()
                    plt.clf()
                    plt.close()
