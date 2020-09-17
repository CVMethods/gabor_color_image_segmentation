import sys
sys.path.append('../')
from hdf5_datasets_generator.berkeley_images_dataset_generator import *
from hdf5_datasets_generator.berkeley_gabor_features_dataset_generator import *
from hdf5_datasets_generator.berkeley_superpixels_dataset_generator import *
from hdf5_datasets_generator.berkeley_graph_gradient_slic_dataset_generator import *
from slic_level_contours.supervised_graph_gradient.train_learning_models import *
from slic_level_contours.supervised_graph_gradient.test_learning_models import *
from slic_level_contours.slic_graph_gradient_to_img_contours import *


num_imgs = 7

periods = [(2., 25.), (2., 45.)]
bandwidths = [(0.7, 30), (1.0, 45)]
crossing_points = [(0.75, 0.75), (0.9, 0.9)]
deviations = [3.0]

n_slic = 500 * 4

# Graph function parameters
graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'eps'

# Distance parameter
similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

generate_h5_images_dataset(num_imgs)
generate_h5_features_dataset(num_imgs, periods, bandwidths, crossing_points, deviations)
generate_h5_superpixels_dataset(num_imgs, n_slic)
generate_h5_graph_gradients_dataset(num_imgs, n_slic, graph_type, similarity_measure)
train_models(num_imgs, n_slic, graph_type, similarity_measure)
test_models(num_imgs, n_slic, graph_type, similarity_measure)

gradients_dir = 'predicted_gradients'
bsd_subset = 'test'
generate_imgcontours_from_graphs(num_imgs, n_slic,graph_type,similarity_measure,gradients_dir, bsd_subset)

gradients_dir = 'gradients'  # 'predicted_gradients'
bsd_subset = 'test'
generate_imgcontours_from_graphs(num_imgs, n_slic,graph_type,similarity_measure,gradients_dir, bsd_subset)

