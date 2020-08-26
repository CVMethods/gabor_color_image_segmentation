import sys
import h5py

from pathlib import Path
sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *

if __name__ == '__main__':
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
    all_f_scores = []
    for features_input_file in input_files:
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = np.array(features_file["/gabor_features"])
            feature_shapes = np.array(features_file["/feature_shapes"])

            n_freq = features_file.attrs['num_freq']
            n_angles = features_file.attrs['num_angles']

            gabor_features_norm = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1], n_freq * n_angles, shape[2])) for features, shape in
                zip(feature_vectors, img_shapes))

            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            # Compute ground distance matrix
            ground_distance = cost_matrix_texture(n_freq, n_angles)

            # Superpixels function parameters
            n_regions = 500 * 4
            convert2lab = True

            # Graph function parameters
            graph_type = 'knn'  # Choose: 'complete', 'knn', 'rag'
            kneighbors = 8
            radius = 10

            # Segmentation parameters
            method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence
            aff_norm_method = 'global'  # Choose: 'global' or 'local'
            graph_mode = 'complete'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
            num_clusters = 'min'

            metrics_values = []
            for im_file, img, g_energies in zip(img_ids, images, gabor_features_norm):
                time_total = time.time()

                print('##############################', im_file, '##############################')

                ''' Computing superpixel regions '''
                regions_slic = slic_superpixel(img, n_regions, convert2lab)

                ''' Computing Graph '''
                graph_raw = get_graph(img, regions_slic, graph_type, kneighbors, radius)

                ''' Updating edges weights with similarity measure (OT/KL) '''
                g_energies_sum = np.sum(g_energies, axis=-1)
                graph_weighted = update_edges_weight(regions_slic, graph_raw, g_energies_sum, ground_distance, method)

                if graph_mode == 'complete':
                    weights = nx.get_edge_attributes(graph_weighted, 'weight').values()
                elif graph_mode == 'mst':
                    # Compute Minimum Spanning Tree
                    graph_mst = get_mst(graph_weighted)
                    weights = nx.get_edge_attributes(graph_mst, 'weight').values()
                    graph_weighted = graph_mst

                ''' Getting number of cluster based on ground truth segmentations '''
                groundtruth_segments = np.array(get_segment_from_filename(im_file))
                n_clusters = get_num_segments(groundtruth_segments)

                if num_clusters == 'max':
                    k = int(n_clusters[0])
                elif num_clusters == 'min':
                    k = int(n_clusters[1])
                elif num_clusters == 'mean':
                    k = int(n_clusters[2])
                elif num_clusters == 'hmean':
                    k = int(n_clusters[3])

                ''' Performing Spectral Clustering on weighted graph '''
                aff_matrix, graph_normalized = distance_matrix_normalization(graph_weighted, weights, aff_norm_method, regions_slic)

                t0 = time.time()
                segmentation = SpectralClustering(n_clusters=15, assign_labels='discretize', affinity='precomputed',
                                                  n_init=100, n_jobs=-1).fit(aff_matrix)
                t1 = time.time()
                print(' Computing time: %.2fs' % (t1 - t0))
                regions_spec = get_sgmnt_regions(graph_weighted, segmentation.labels_, regions_slic)

                ''' Evaluation of segmentation'''
                if len(np.unique(regions_spec)) == 1:
                    metrics_values.append((0., 0.))
                else:
                    m = metrics(None, regions_spec, groundtruth_segments)
                    m.set_metrics()
                    # m.display_metrics()
                    vals = m.get_metrics()
                    metrics_values.append((vals['recall'], vals['precision']))

                ##############################################################################
                '''Visualization Section: show and/or save images'''
                # General Params
                save_fig = True
                fontsize = 10
                file_name = im_file

                outdir = '../outdir/' + \
                         'slic_level_segmentation/' + \
                         num_imgs_dir + \
                         'spectral_clustering/' + \
                         method + '/' + \
                         graph_type + '_graph/' + \
                         features_input_file[:-3] + '/' + \
                         aff_norm_method + '_normalization/' + \
                         graph_mode + '_graph/' +\
                         'computation_support/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                # Show Input image
                fig_title = 'Input Image'
                show_and_save_img(img, fig_title, fontsize, save_fig, outdir, file_name)

                # Show SLIC result
                fig_title = 'Superpixel Regions'
                img_name = '_slic'
                show_and_save_regions(img, regions_slic, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                # Show Graph with uniform weight
                fig_title = 'Graph (' + graph_type + ')'
                img_name = '_raw_' + graph_type
                colbar_lim = (0, 1)
                show_and_save_imgraph(img, regions_slic, graph_raw, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

                # Show Graph with updated weights
                fig_title = graph_mode + ' Weighted Graph (' + graph_type + ')'
                img_name = '_weighted_' + graph_type
                colbar_lim = (min(weights), max(weights))
                show_and_save_imgraph(img, regions_slic, graph_weighted, fig_title, img_name, fontsize, save_fig, outdir, file_name, colbar_lim)

                # fig_title = 'Spectral Graph'
                # img_name = '_spec_graph'
                # show_and_save_spectralgraph(graph, fig_title, img_name, fontsize, save_fig, outdir, file_name)

                fig_title = 'Affinity Matrix'
                img_name = '_aff_mat'
                show_and_save_affmat(aff_matrix, fig_title, img_name, fontsize, save_fig, outdir, file_name)
                ##############################################################################
                # Segmentation results visualization
                outdir = '../outdir/' + \
                         'slic_level_segmentation/' + \
                         num_imgs_dir + \
                         'spectral_clustering/' + \
                         method + '/' + \
                         graph_type + '_graph/' + \
                         features_input_file[:-3] + '/' + \
                         aff_norm_method + '_normalization/' + \
                         graph_mode + '_graph/' + \
                         'results/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                fig_title = 'Spectral Clustering Result k=%d' % k
                fig_label = (vals['recall'], vals['precision'], (t1 - t0))
                img_name = '_spec_result'
                show_and_save_result(img, regions_spec, fig_title, fig_label, img_name, fontsize, save_fig, outdir, file_name)

                plt.close('all')

            metrics_values = np.array(metrics_values)
            recall = metrics_values[:, 0]
            precision = metrics_values[:, 1]
            f_score = hmean((precision, recall), axis=0)
            all_f_scores.append(f_score)

            plt.figure(dpi=180)
            plt.plot(np.arange(len(image_vectors)) + 1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(image_vectors)) + 1, precision, '-o', c='r', label='precision')
            plt.title('Spectral clustering P/R histogram')
            plt.xlabel('Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                    recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                    precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'Spectral_clustering_PR_hist.png', bbox_inches='tight')

            plt.figure(dpi=180)
            sns.distplot(recall, color='black', label='recall')
            sns.distplot(precision, color='red', label='precision')
            plt.title('Spectral clustering P/R density histogram')
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'Spectral_clustering_PR_density_hist.png', bbox_inches='tight')

            plt.figure(dpi=180)
            ax = plt.gca()
            ax.boxplot(list([precision, recall]))
            ax.set_title('Spectral clustering P/R density box plot')
            ax.set_xticklabels(['precision', 'recall'])
            ax.set_xlabel('F-score: %.3f' % np.median(f_score))
            plt.grid()
            plt.savefig(outdir + 'Spectral_clustering_PR_boxplot.png', bbox_inches='tight')

            plt.close('all')

    all_f_scores = np.array(all_f_scores)
    index = np.argsort(np.median(all_f_scores, axis=1))
    input_files = np.array(input_files)

    outdir = '../outdir/' + \
        'slic_level_segmentation/' + \
        num_imgs_dir + \
        'spectral_clustering/' + \
        method + '/' + \
        graph_type + '_graph/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    plt.figure(dpi=180)
    ax = plt.gca()
    ax.boxplot(all_f_scores[index].T, vert=False)
    ax.set_title('Spectral clustering F scores: ' + aff_norm_method + ' norm, ' + graph_mode + ' graph')
    # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
    ax.set_yticklabels(input_files[index], fontsize=5)
    plt.grid()
    plt.savefig(outdir + 'Spectral_clustering_Fscores_' + aff_norm_method + '_' + graph_mode + '_boxplot.png', bbox_inches='tight')
