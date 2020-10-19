import sys
sys.path.append('../')
from hdf5_datasets_generator.berkeley_images_dataset_generator import *
from hdf5_datasets_generator.berkeley_gabor_features_dataset_generator import *
from hdf5_datasets_generator.berkeley_superpixels_dataset_generator import *
from hdf5_datasets_generator.berkeley_graph_gradient_slic_dataset_generator import *
from hdf5_datasets_generator.berkeley_graph_gradient_dataset_generator import *
from slic_level_contours.supervised_graph_gradient.train_test_learning_models import *
from slic_level_contours.supervised_graph_gradient.test_learning_models import *
from slic_level_contours.slic_graph_gradient_to_img_contours import *
from slic_level_segmentation.threshold_graphcut_hdf5_dataset import *
from slic_level_segmentation.spectral_clustering_hdf5_dataset import *
from slic_level_segmentation.normalized_graphcut_hdf5_dataset import *
from slic_level_segmentation.affinity_propagation_hdf5_dataset import *


num_imgs = 500

# Gabor filter parameters
periods = [(3., 40.)]  #(2., 25.),, (2., 20.), (4., 45.), (4., 20.)
bandwidths = [(0.7, 30)]  #, (1.0, 45), (0.7, 30)
crossing_points = [(0.9, 0.9)]  #(0.65, 0.65),
deviations = [3.]

# Graph function parameters
kneighbors = 4
# Distance parameter
similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

generate_h5_images_dataset(num_imgs)
generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations)
generate_h5_graph_gradients_dataset(num_imgs, kneighbors, similarity_measure)
train_test_models(num_imgs, similarity_measure, kneighbors, None, None)

gradients_dir = 'predicted_gradients'
generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, kneighbors=kneighbors)

########################################################################################################################

# gradients_dir = 'gradients'
# generate_imgcontours_from_graphs(num_imgs, similarity_measure, gradients_dir, kneighbors=kneighbors)
