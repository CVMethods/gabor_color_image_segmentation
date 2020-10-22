import sys
sys.path.append('../')
from hdf5_datasets_generator.berkeley_images_dataset_generator import *
from hdf5_datasets_generator.berkeley_gabor_features_dataset_generator import *
from hdf5_datasets_generator.berkeley_superpixels_dataset_generator import *
from hdf5_datasets_generator.berkeley_graph_gradient_slic_dataset_generator import *
from slic_level_contours.supervised_graph_gradient.train_test_learning_models import *
from slic_level_contours.supervised_graph_gradient.test_learning_models import *
from slic_level_contours.slic_graph_gradient_to_img_contours import *
from slic_level_segmentation.threshold_graphcut_hdf5_dataset import *
from slic_level_segmentation.spectral_clustering_hdf5_dataset import *
from slic_level_segmentation.normalized_graphcut_hdf5_dataset import *
from slic_level_segmentation.affinity_propagation_hdf5_dataset import *

''' Generate images dataset'''
num_imgs = 7
generate_h5_images_dataset(num_imgs)

# ''' Generate Gabor features datasets'''
# periods = [(2., 25.), (3., 25.), (2., 50.), (3., 50.)]
# bandwidths = [(1, 30)]
# crossing_points = [(0.9, 0.9)]
# deviations = [3.]
#
# generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations)


'''Generate image superpixels datasets'''
n_slic_base = 500

# Graph function parameters
graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'eps'
# Distance parameter
similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence
# Directory parameter
gradients_dir = 'predicted_gradients'

for ns in [3, 7, 11]:#[3, 5, 7, 9, 11]
    # Superpixel parameters
    n_slic = n_slic_base * ns

    print('\n Computing %d Superpixels dataset \n' % n_slic)
    generate_h5_superpixels_dataset(num_imgs, n_slic)

    print('\n Computing %d slic Graph Gradients dataset \n' % n_slic)
    generate_h5_slic_graph_gradients_dataset(num_imgs, n_slic, graph_type, similarity_measure)

    print('\n Computing %d slic learning stage' % n_slic)
    train_test_models(num_imgs, similarity_measure, None, n_slic, graph_type)

    print('\n Computing %d slic image boundaries \n' % n_slic)
    generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, n_slic=n_slic, graph_type=graph_type)

    ###################################################################################################################
    # '''Segmentation from slic graph gradient'''
    #
    # # Threshold graph cut parameters
    # graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
    # law_type = 'log'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
    # cut_level = 0.8  # set threshold at the 90% quantile level
    #
    # print('\n Computing %d slic threshold graph cut \n' % n_slic)
    # threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode, law_type,
    #                                 cut_level)

    # # Spectral clustering and normalized cuts parameters
    # graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
    # aff_norm_method = 'global'  # Choose: 'global' or 'local'
    # num_clusters = 'min'
    #
    # print('\n Computing %d slic spectral clustering \n' % n_slic)
    # spectral_clustering_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
    #                                  aff_norm_method, num_clusters)
    #
    # print('\n Computing %d slic normalized cuts \n' % n_slic)
    # normalized_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
    #                                  aff_norm_method)
    #
    # print('\n Computing %d slic affinity propagation \n' % n_slic)
    # affinity_propagation_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
    #                                   aff_norm_method)

#######################################################################################################################

# gradients_dir = 'gradients'
# generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, n_slic=n_slic, graph_type=graph_type)
#
# # Segmentation parameters
# graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
# law_type = 'log'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
# cut_level = 0.9  # set threshold at the 90% quantile level
#
# threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode, law_type,
#                                 cut_level)
#
# # Segmentation parameters
# graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
# aff_norm_method = 'global'  # Choose: 'global' or 'local'
# num_clusters = 'min'
#
# spectral_clustering_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
#                                  aff_norm_method, num_clusters)
#
# normalized_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
#                                  aff_norm_method)
#
# # affinity_propagation_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
# #                                   aff_norm_method)
