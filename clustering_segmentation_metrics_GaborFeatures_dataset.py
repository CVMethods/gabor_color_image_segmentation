"""
=========================================================
Comparing different clustering algorithms on images
=========================================================

Ce script est inspiré de comparaison de différentes techniques de clustering en scikit-learn et adapté aux images. Le clustering, très basique, se fait sur la base de couleurs RGB.

Il n'y a que deux algorithmes assez rapide pour marcher sur les pixels. Cependant, d'autres clustering techniques peuvent être testés après sous-échantillonnage de l'image d'entrée. Il faudra tester ces algorithmes sur l'aggrégation de superpixels.

Pour certain, il faut spécifier le nombre de clusters.
"""
from joblib import Parallel, delayed
from scipy.cluster import vq
from sklearn.decomposition import PCA

print(__doc__)

import time, warnings, pdb, os

import numpy as np
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skimage.io import imread
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from petastorm import make_reader

from PIL import Image
from matplotlib import pyplot as plt
from BSD_metrics.metrics import *


def clustering_segmentation_and_metrics(i_dataset, dataset, algo_params, num_clusters):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    img_id, X, y, n_clusters, img_size = dataset

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
        ax = plt.gca()
        im = ax.imshow(y_pred, cmap=plt.cm.get_cmap('jet', nc))
        ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                       length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
        ax.set_title(algo_name + ' k=%d' % nc, fontsize=10)
        ax.set_xlabel(('Recall: %.3f, Precision: %.3f, Time: %.2fs' % (
            metrics_values['recall'], metrics_values['precision'], (t1 - t0))).lstrip('0'), fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax, ticks=(np.arange(nc) + 0.5) * (nc - 1) / nc)
        cb.set_label(label='$labels$', fontsize=10)
        cb.ax.tick_params(labelsize=6)
        cb.ax.set_yticklabels([r'${{{}}}$'.format(val) for val in range(nc)])
        plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_' + algo_name + '_' + num_clusters + '_segm.png')

        plt.close('all')

        img_metrics.append(metrics_values)
        segmentations.append(y_pred)

    # return segmentations, img_metrics
    return img_metrics


if __name__ == '__main__':

    np.random.seed(0)

    dataset_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/complete/Berkeley_GaborFeatures_6freq_6ang'

    datasets = []
    with make_reader(dataset_url) as reader:
        for sample in reader:
            datasets.append(((sample.img_id.decode('UTF-8'), sample.gabor_features, sample.ground_truth, sample.num_seg,
                              sample.img_shape), {}))

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 4,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1,
                    'n_jobs': -1}

    num_cores = -1

    possible_num_clusters = ['max', 'min', 'mean', 'hmean', 'const']
    for num_clusters in possible_num_clusters:

        outdir = 'data/outdir/' + dataset_url[-10:] + '/' + num_clusters + '_nclusters/'

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        segmentation_metrics = Parallel(n_jobs=num_cores, prefer='processes')(
            delayed(clustering_segmentation_and_metrics)(i_dataset+1, dataset, algo_params, num_clusters) for i_dataset, (dataset, algo_params) in enumerate(datasets))

        KMeans_metrics = []
        MiniBatchKMeans_metrics = []
        Birch_metrics = []
        GaussianMixture_metrics = []

        for ii in range(len(datasets)):
            algo_metrics = segmentation_metrics[ii]
            KMeans_metrics.append((algo_metrics[0]['recall'], algo_metrics[0]['precision']))
            MiniBatchKMeans_metrics.append((algo_metrics[1]['recall'], algo_metrics[1]['precision']))
            Birch_metrics.append((algo_metrics[2]['recall'], algo_metrics[2]['precision']))
            GaussianMixture_metrics.append((algo_metrics[3]['recall'], algo_metrics[3]['precision']))

        KMeans_metrics = np.array(KMeans_metrics)
        MiniBatchKMeans_metrics = np.array(MiniBatchKMeans_metrics)
        Birch_metrics = np.array(Birch_metrics)
        GaussianMixture_metrics = np.array(GaussianMixture_metrics)

        algorithms_metrics = [KMeans_metrics, MiniBatchKMeans_metrics, Birch_metrics, GaussianMixture_metrics]
        algorithms_names = ['KMeans', 'MiniBatchKMeans', 'Birch', 'GaussianMixture']

        for name, result_metrics in zip(algorithms_names, algorithms_metrics):
            recall = result_metrics[:, 0]
            precision = result_metrics[:, 1]

            plt.figure(dpi=180)
            plt.plot(np.arange(len(datasets))+1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(datasets)) + 1, precision, '-o', c='r', label='precision')
            plt.title(name + ' P/R histogram ' + num_clusters + ' nclusters')
            plt.xlabel('Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(), precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + name + '_PR_hist_' + num_clusters + '_nclusters.png', bbox_inches='tight')

            plt.close('all')