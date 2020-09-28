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


num_imgs = 7

periods = [(2., 25.)]  #(2., 25.),, (2., 20.), (4., 45.), (4., 20.)
bandwidths = [(1., 45)]  #, (1.0, 45), (0.7, 30)
crossing_points = [(0.9, 0.9)]  #(0.65, 0.65),
deviations = [3.]

n_slic = 500 * 4

# Graph function parameters
graph_type = '8nn'  # Choose: 'complete', 'knn', 'rag', 'eps'

# Distance parameter
similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

generate_h5_images_dataset(num_imgs)
generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations)
generate_h5_superpixels_dataset(num_imgs, n_slic)
generate_h5_graph_gradients_dataset(num_imgs, n_slic, graph_type, similarity_measure)
train_test_models(num_imgs, n_slic, graph_type, similarity_measure)

gradients_dir = 'predicted_gradients'
bsd_subset = 'test'
generate_imgcontours_from_graphs(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset)

gradients_dir = 'gradients'
bsd_subset = 'all'
generate_imgcontours_from_graphs(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset)

# Segmentation parameters
graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
law_type = 'log'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
cut_level = 0.9  # set threshold at the 90% quantile level
gradients_dir = 'predicted_gradients'
bsd_subset = 'test'

threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset, graph_mode, law_type, cut_level)

# Segmentation parameters
graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
law_type = 'log'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
cut_level = 0.9  # set threshold at the 90% quantile level
gradients_dir = 'gradients'
bsd_subset = 'test'

threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset, graph_mode, law_type, cut_level)

# Segmentation parameters
graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
aff_norm_method = 'global'  # Choose: 'global' or 'local'
num_clusters = 'min'
gradients_dir = 'predicted_gradients'
bsd_subset = 'test'

spectral_clustering_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset,
                                graph_mode, aff_norm_method, num_clusters)

# Segmentation parameters
graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
aff_norm_method = 'global'  # Choose: 'global' or 'local'
num_clusters = 'min'
gradients_dir = 'gradients'
bsd_subset = 'test'

spectral_clustering_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset,
                                graph_mode, aff_norm_method, num_clusters)


# Segmentation parameters
graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
aff_norm_method = 'global'  # Choose: 'global' or 'local'
gradients_dir = 'predicted_gradients'
bsd_subset = 'test'

normalized_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset,
                                graph_mode, aff_norm_method)

# Segmentation parameters
graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
aff_norm_method = 'global'  # Choose: 'global' or 'local'
gradients_dir = 'gradients'
bsd_subset = 'test'

normalized_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset,
                                graph_mode, aff_norm_method)
