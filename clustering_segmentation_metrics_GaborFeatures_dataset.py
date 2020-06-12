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


def clustering_segmentation_and_metrics(dataset, algo_params):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    img_id, X, y, n_clusters, img_size = dataset

    print('dataset:', img_id)

    # img_size = img_size[:-1]
    rows, cols, channels = img_size
    params['n_clusters'] = int(n_clusters[1])
    # print(rows, cols, params['n_clusters'])

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
            # if algo_name is 'GaussianMixture' or 'Birch':
            #     X = X_reduced
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

        # plt.figure(dpi=180)
        # ax = plt.gca()
        # im = ax.imshow(y_pred, cmap=plt.cm.get_cmap('jet', nc))
        # ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
        #                length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
        # ax.set_title(algo_name + ' k=%d' % nc, fontsize=10)
        # ax.set_xlabel(('Recall: %.3f, Precision: %.3f, Time: %.2fs' % (
        #     metrics_values['recall'], metrics_values['precision'], (t1 - t0))).lstrip('0'), fontsize=10)
        #
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cb = plt.colorbar(im, cax=cax, ticks=(np.arange(nc) + 0.5) * (nc - 1) / nc)
        # cb.set_label(label='$labels$', fontsize=10)
        # cb.ax.tick_params(labelsize=6)
        # cb.ax.set_yticklabels([r'${{{}}}$'.format(val) for val in range(nc)])
        # plt.savefig(outdir + img_id + '_' + algo_name + '_segm.png')
        # plt.clf()
        # plt.cla()

        img_metrics.append(metrics_values)
        segmentations.append(y_pred)

    return segmentations, img_metrics


if __name__ == '__main__':

    np.random.seed(0)

    # img_path = "data/myFavorite_BSDimages/"
    dataset_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/test/Berkeley_GaborFeatures'
    outdir = 'data/outdir/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    datasets = []
    with make_reader(dataset_url) as reader:
        for sample in reader:
            datasets.append(((sample.img_id.decode('UTF-8'), sample.gabor_features, sample.ground_truth, sample.mean_max_min_nseg, sample.img_shape), {}))

    default_base = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': 10,
                        'min_samples': 20,
                        'xi': 0.05,
                        'min_cluster_size': 0.1,
                        'n_jobs': -1}

    num_cores = -1
    segmentation_metrics = Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(clustering_segmentation_and_metrics)(dataset, algo_params) for dataset, algo_params in datasets)


    pdb.set_trace()
    # plt.show()
