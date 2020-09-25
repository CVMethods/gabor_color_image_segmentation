import sys
import h5py

from pathlib import Path

sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *


def get_img_contours(im_file, img, regions_slic, graph_raw, perceptual_gradients, gradients_dir, outdir):
    print('##############################', im_file, '##############################')

    graph_weighted = graph_raw.copy()
    ''' Updating edges weights with optimal transport '''
    if gradients_dir == 'gradients':
        perceptual_gradients = np.sum(perceptual_gradients[:, :-1], axis=-1)

    perceptual_gradients = (perceptual_gradients - min(perceptual_gradients)) / (
                max(perceptual_gradients) - min(perceptual_gradients)) * 255

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_weighted[e[0]][e[1]]['weight'] = perceptual_gradients[i_edge]

    img_grad = graph2gradient(img, graph_weighted, perceptual_gradients, regions_slic)

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    img = Image.fromarray(np.uint8(img_grad))
    img.save(outdir + im_file + '.png')
    # plt.imsave(outdir + im_file + '.png', img_grad, cmap='gray')


def generate_imgcontours_from_graphs(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset):
    num_cores = -1

    hdf5_indir_im = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    hdf5_indir_spix = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'superpixels/' +
                           str(n_slic) + '_slic')
    hdf5_indir_grad = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + gradients_dir + '/' +
                           str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure)

    num_imgs_dir = str(num_imgs) + 'images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = np.array(images_file["/images"])
    img_shapes = np.array(images_file["/image_shapes"])
    img_ids = np.array(images_file["/image_ids"])
    img_subdirs = np.array(images_file["/image_subdirs"])

    superpixels_file = h5py.File(hdf5_indir_spix / "Berkeley_superpixels.h5", "r+")
    superpixels_vectors = np.array(superpixels_file["/superpixels"])

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    superpixels = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img_spix, (shape[0], shape[1])) for img_spix, shape in
        zip(superpixels_vectors, img_shapes)))

    n_regions = superpixels_file.attrs['num_slic_regions']
    n_slic_regions = str(superpixels_file.attrs['num_slic_regions']) + '_regions'

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    if bsd_subset == 'test' or gradients_dir == 'predicted_gradients':
        test_indices = []
        for ii in range(len(images)):
            if img_subdirs[ii] == 'test':  # Need to change the name of directory to add the gradients to training dataset
                test_indices.append(ii)

        img_ids = img_ids[test_indices]
        images = images[test_indices]
        superpixels = superpixels[test_indices]

    ''' Computing Graphs for test set images'''
    raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
        zip(images, superpixels))

    if gradients_dir == 'gradients':

        input_directories = sorted(os.listdir(hdf5_indir_grad))

        for gradients_input_dir in input_directories:
            with h5py.File(hdf5_indir_grad / gradients_input_dir / 'gradients.h5', "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_dir)
                t0 = time.time()

                gradient_vectors = np.array(gradients_file["/perceptual_gradients"])
                gradient_shapes = np.array(gradients_file["/gradient_shapes"])

                if bsd_subset == 'test':
                    gradient_vectors = gradient_vectors[test_indices]
                    gradient_shapes = gradient_shapes[test_indices]

                imgs_gradients = Parallel(n_jobs=num_cores)(
                    delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in
                    zip(gradient_vectors, gradient_shapes))

                outdir = '../outdir/' + \
                         num_imgs_dir + \
                         'image_contours/' + \
                         (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                         'SimpleSum' + '_' + bsd_subset + '/' + \
                         gradients_input_dir + '/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                Parallel(n_jobs=num_cores)(
                    delayed(get_img_contours)(im_file, img, regions_slic, graph_raw, perceptual_gradients, gradients_dir, outdir)
                    for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                    zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))

    if gradients_dir == 'predicted_gradients':

        model_input_dirs = sorted(os.listdir(hdf5_indir_grad))
        for mm, model_name in enumerate(model_input_dirs):
            input_directories = sorted(os.listdir(hdf5_indir_grad / model_name))

            for gradients_input_dir in input_directories:
                with h5py.File(hdf5_indir_grad / model_name / gradients_input_dir / 'predicted_gradients.h5',
                               "r+") as gradients_file:
                    print('Reading Berkeley features data set')
                    print('File name: ', gradients_input_dir)
                    t0 = time.time()

                    imgs_gradients = np.array(gradients_file["/predicted_gradients"])

                    outdir = '../outdir/' + \
                             num_imgs_dir + \
                             'image_contours/' + \
                             (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                             model_name + '/' + \
                             gradients_input_dir + '/'

                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    _ = Parallel(n_jobs=num_cores)(
                        delayed(get_img_contours)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                  gradients_dir, outdir)
                        for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                        zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))


if __name__ == '__main__':
    num_imgs = 7

    n_slic = 500 * 2

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    gradients_dir = 'predicted_gradients'  # 'predicted_gradients'
    bsd_subset = 'all'

    generate_imgcontours_from_graphs(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset)
