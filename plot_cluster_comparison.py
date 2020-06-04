"""
=========================================================
Comparing different clustering algorithms on images
=========================================================

Ce script est inspiré de comparaison de différentes techniques de clustering en scikit-learn et adapté aux images. Le clustering, très basique, se fait sur la base de couleurs RGB.

Il n'y a que deux algorithmes assez rapide pour marcher sur les pixels. Cependant, d'autres clustering techniques peuvent être testés après sous-échantillonnage de l'image d'entrée. Il faudra tester ces algorithmes sur l'aggrégation de superpixels.

Pour certain, il faut spécifier le nombre de clusters.
"""
print(__doc__)

import time, warnings, pdb, os

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from skimage.io import imread
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from PIL import Image
from matplotlib import pyplot as plt

np.random.seed(0)

img_names = ['eagle.png', 'example_01.jpg', 'example_02.jpg', 'example_03.jpg', 'example_04.jpg', 'example_05.jpg',
             'example_06.jpg']
clusters = [2, 5, 3, 5, 3, 5, 5]

ss = 1
datasets = []
for ii in range(len(img_names)):
    img = imread('images/' + img_names[ii])
    xx, yy = img.size
    img = img.resize((xx // ss, yy // ss))
    img = np.array(img)[:, :, 0:3]
    yy, xx, zz = img.shape
    X, y, img_size = img.reshape((xx * yy, zz)), None, (yy, xx)
    datasets.append(((X, y, clusters[ii], img_size), {}))

# ============
# Set up cluster parameters
# ============

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 2,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

# datasets = [
#     (noisy_circles, {'damping': .77, 'preference': -240,
#                      'quantile': .2, 'n_clusters': 2,
#                      'min_samples': 20, 'xi': 0.25}),
#     (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
#     (varied, {'eps': .18, 'n_neighbors': 2,
#               'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
#     (aniso, {'eps': .15, 'n_neighbors': 2,
#              'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
#     (blobs, {}),
#     (no_structure, {})]


for i_dataset, (dataset, algo_params) in enumerate(datasets):

    plt.figure(i_dataset + 1, figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.1)
    plot_num = 1

    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    print('dataset %d' % i_dataset)

    X, y, n_clusters, img_size = dataset
    params['n_clusters'] = n_clusters

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # # estimate bandwidth for mean shift
    # bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    # ward = cluster.AgglomerativeClustering(
    #     n_clusters=params['n_clusters'], linkage='ward',
    #     connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'],
                            xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    # uncomment to try other clustering algorithms
    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        # ('AffinityPropagation', affinity_propagation), # commented out since excessively long
        # ('MeanShift', ms), # commented out since too long bandwidth computation
        # ('SpectralClustering', spectral), # commented cause difficult to have it working
        # ('Ward', ward), # commented out since too long
        # ('AgglomerativeClustering', average_linkage),
        # ('DBSCAN', dbscan), # commented out since memory consuming
        # ('OPTICS', optics), # commented out since too long
        ('Birch', birch),
        # ('GaussianMixture', gmm) # commented out since too long on big images
    )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        print('algorithm %s' % name)

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

        plt.subplot(2, (len(clustering_algorithms) // 2), plot_num)
        plt.title(name, size=10)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])

        plt.imshow(y_pred.reshape(*img_size))
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

    # plt.show(block=False)
    plt.savefig(os.path.splitext(img_names[i_dataset])[0] + '_segm.png')

plt.show()
