import sys
import h5py

from pathlib import Path
sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


def get_thr_graphcut_metrics(im_file, img, regions_slic, graph_raw, perceptual_gradients, graph_mode, law_type, cut_level, gradients_dir, outdir):
    print('##############################', im_file, '##############################')

    graph_weighted = graph_raw.copy()
    ''' Updating edges weights with optimal transport '''
    if gradients_dir == 'gradients':
        perceptual_gradients = np.sum(perceptual_gradients[:, :-1], axis=-1)

    perceptual_gradients = (perceptual_gradients - min(perceptual_gradients)) / (max(perceptual_gradients) - min(perceptual_gradients))

    for i_edge, e in enumerate(list(graph_raw.edges)):
        graph_weighted[e[0]][e[1]]['weight'] = perceptual_gradients[i_edge]

    if graph_mode == 'complete':
        weights = nx.get_edge_attributes(graph_weighted, 'weight').values()
    elif graph_mode == 'mst':
        # Compute Minimum Spanning Tree
        graph_mst = get_mst(graph_weighted)
        weights = nx.get_edge_attributes(graph_mst, 'weight').values()
        graph_weighted = graph_mst

    ''' Performing Graph cut on weighted RAG '''
    thresh, params = fit_distribution_law(list(weights), cut_level, law_type)

    t0 = time.time()
    graph_aftercut = graph_weighted.copy()
    graph.cut_threshold(regions_slic, graph_aftercut, thresh, in_place=True)
    t1 = time.time()
    print(' Computing time: %.2fs' % (t1 - t0))

    regions_aftercut = graph2regions(graph_aftercut, regions_slic)

    # Generating graph after the 1st cut
    graph_mean_color = graph.rag_mean_color(img, regions_aftercut, mode='distance')

    regions_aftermerge = graph.merge_hierarchical(regions_aftercut, graph_mean_color, thresh=70, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

    ''' Evaluation of segmentation'''
    groundtruth_segments = np.array(get_segment_from_filename(im_file))

    if len(np.unique(regions_aftercut)) == 1:
        # metrics_values.append((0., 0.))
        metrics_vals = (0., 0.)
    else:
        m = metrics(None, regions_aftercut, groundtruth_segments)
        m.set_metrics()
        # m.display_metrics()
        vals = m.get_metrics()
        # metrics_values.append((vals['recall'], vals['precision']))
        metrics_vals = (vals['recall'], vals['precision'])

    if len(np.unique(regions_aftermerge)) == 1:
        # metrics_values.append((0., 0.))
        metrics_vals1 = (0., 0.)
    else:
        m = metrics(None, regions_aftermerge, groundtruth_segments)
        m.set_metrics()
        # m.display_metrics()
        vals = m.get_metrics()
        # metrics_values.append((vals['recall'], vals['precision']))
        metrics_vals1 = (vals['recall'], vals['precision'])

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    # General Params
    save_fig = True
    fontsize = 10
    file_name = im_file

    output_dir = outdir + 'computation_support/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Show Graph with updated weights
    fig_title = graph_mode + ' Weighted Graph'
    img_name = '_weighted_' + graph_mode + 'graph'
    colbar_lim = (min(weights), max(weights))
    show_and_save_imgraph(img, regions_slic, graph_weighted, fig_title, img_name, fontsize, save_fig, output_dir, file_name,
                          colbar_lim)

    # # Show one SLIC region and its neighbors
    # region = 109
    # show_and_save_some_regions(img, regions, region, rag, save_fig, outdir, file_name)

    # Edges weight distribution
    fig_title = 'Edges Weight Distribution' + law_type
    img_name = '_weight_dist'
    show_and_save_dist(weights, law_type, thresh, params, fig_title, img_name, fontsize, save_fig, output_dir, file_name)

    # RAG after cut
    fig_title = 'RAG after cut'
    img_name = '_thr_graph_aftercut'
    colbar_lim = (min(weights), max(weights))
    show_and_save_imgraph(img, regions_slic, graph_aftercut, fig_title, img_name, fontsize, save_fig,
                          output_dir, file_name, colbar_lim)

    # RAG mean color after cut
    fig_title = 'RAG mean color after cut'
    img_name = '_graph_mean_color'
    colbar_lim = (None, None)
    show_and_save_imgraph(img, regions_aftercut, graph_mean_color, fig_title, img_name, fontsize, save_fig,
                          output_dir, file_name, colbar_lim)

    ##############################################################################
    # Segmentation results visualization
    output_dir = outdir + 'results/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fig_title = 'Segmentation Result '
    fig_label = (metrics_vals[0], metrics_vals[1], (t1 - t0))
    img_name = '_graphcut_result'
    show_and_save_result(img, regions_aftercut, fig_title, fig_label, img_name, fontsize, save_fig, output_dir, file_name)

    fig_title = 'Segmentation Result '
    fig_label = (metrics_vals1[0], metrics_vals1[1], (t1 - t0))
    img_name = '_graphmerge_result'
    show_and_save_result(img, regions_aftermerge, fig_title, fig_label, img_name, fontsize, save_fig, output_dir, file_name)

    return metrics_vals1


def threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
                                    law_type, cut_level):
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
    image_vectors = images_file["images"][:]
    img_shapes = images_file["image_shapes"][:]
    img_ids = images_file["image_ids"][:]
    img_subdirs = images_file["image_subdirs"][:]

    superpixels_file = h5py.File(hdf5_indir_spix / "Berkeley_superpixels.h5", "r+")
    superpixels_vectors = superpixels_file["superpixels"][:]

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    superpixels = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img_spix, (shape[0], shape[1])) for img_spix, shape in
        zip(superpixels_vectors, img_shapes)))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    ''' Computing Graphs for test set images'''
    raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
        zip(images, superpixels))

    if gradients_dir == 'predicted_gradients':
        test_indices = []
        for ii in range(len(images)):
            if img_subdirs[ii] == 'test':
                test_indices.append(ii)

        img_ids = img_ids[test_indices]
        images = images[test_indices]
        superpixels = superpixels[test_indices]

        model_input_dirs = sorted(os.listdir(hdf5_indir_grad))
        for mm, model_name in enumerate(model_input_dirs):
            input_files = os.listdir(hdf5_indir_grad / model_name)
            all_f_scores = []
            all_precisions = []
            all_recalls = []
            for gradients_input_file in input_files:
                with h5py.File(hdf5_indir_grad / model_name / gradients_input_file / 'predicted_gradients.h5',
                               "r+") as gradients_file:
                    print('Reading Berkeley features data set')
                    print('File name: ', gradients_input_file)

                    imgs_gradients = gradients_file["predicted_gradients"][:]

                    outdir = '../outdir/' + \
                             num_imgs_dir + \
                             'slic_level_segmentation/' + \
                             (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                             model_name + '/' + \
                             'threshold_graphcut_' + law_type + '_dist_' + graph_mode + '_graph/' + \
                             gradients_input_file + '/'

                    metrics_values = Parallel(n_jobs=num_cores)(
                        delayed(get_thr_graphcut_metrics)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                          graph_mode, law_type, cut_level, gradients_dir, outdir)
                        for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                        zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))

                    outdir += 'results/'
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    metrics_values = np.array(metrics_values)
                    recall = metrics_values[:, 0]
                    all_recalls.append(recall)

                    precision = metrics_values[:, 1]
                    all_precisions.append(precision)

                    f_score = hmean((precision, recall), axis=0)
                    all_f_scores.append(f_score)

                    plt.figure(dpi=180)
                    plt.plot(np.arange(len(images)) + 1, recall, '-o', c='k', label='recall')
                    plt.plot(np.arange(len(images)) + 1, precision, '-o', c='r', label='precision')
                    plt.title('Thr graphcut P/R histogram')
                    plt.xlabel(
                        'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                            recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                            precision.min(), precision.mean(), np.median(precision), precision.std()))
                    plt.ylim(0, 1.05)
                    plt.legend()
                    plt.grid()
                    plt.savefig(outdir + 'Thr_graphcut_PR_hist.png', bbox_inches='tight')

                    plt.figure(dpi=180)
                    sns.distplot(recall, color='black', label='recall')
                    sns.distplot(precision, color='red', label='precision')
                    plt.title('Thr graphcut P/R density histogram')

                    plt.legend()
                    plt.grid()
                    plt.savefig(outdir + 'Thr_graphcut_PR_density_hist.png', bbox_inches='tight')

                    plt.figure(dpi=180)
                    ax = plt.gca()
                    ax.boxplot(list([precision, recall]))
                    ax.set_title('Thr graphcut P/R density box plot')
                    ax.set_xticklabels(['precision', 'recall'])
                    ax.set_xlabel('F-score: %.3f' % np.median(f_score))
                    plt.grid()
                    plt.savefig(outdir + 'Thr_graphcut_PR_boxplot.png', bbox_inches='tight')

                    plt.close('all')

            outdir = '../outdir/' + \
                     num_imgs_dir + \
                     'slic_level_segmentation/' + \
                     (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                     model_name + '/' + \
                     'threshold_graphcut_' + law_type + '_dist_' + graph_mode + '_graph/' \

            if not os.path.exists(outdir):
                os.makedirs(outdir)

            all_f_scores = np.array(all_f_scores)
            index = np.argsort(np.median(all_f_scores, axis=1))
            input_files = np.array(input_files)

            plt.figure(dpi=180)
            ax = plt.gca()
            ax.boxplot(all_f_scores[index].T, vert=False)
            ax.set_title('Thr graphcut F scores: ' + law_type + ' dist, ' + graph_mode + ' graph')
            # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
            ax.set_yticklabels(input_files[index], fontsize=5)
            plt.grid()
            plt.savefig(outdir + 'Thr_graphcut_Fscores_' + law_type + '_' + graph_mode + '_boxplot.png',
                        bbox_inches='tight')

            all_recalls = np.array(all_recalls)
            index = np.argsort(np.median(all_recalls, axis=1))
            input_files = np.array(input_files)

            plt.figure(dpi=180)
            ax = plt.gca()
            ax.boxplot(all_recalls[index].T, vert=False)
            ax.set_title('Thr graphcut recall: ' + law_type + ' dist, ' + graph_mode + ' graph')
            # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
            ax.set_yticklabels(input_files[index], fontsize=5)
            plt.grid()
            plt.savefig(outdir + 'Thr_graphcut_recalls_' + law_type + '_' + graph_mode + '_boxplot.png',
                        bbox_inches='tight')

            all_precisions = np.array(all_precisions)
            index = np.argsort(np.median(all_precisions, axis=1))
            input_files = np.array(input_files)

            plt.figure(dpi=180)
            ax = plt.gca()
            ax.boxplot(all_precisions[index].T, vert=False)
            ax.set_title('Thr graphcut precision: ' + law_type + ' dist, ' + graph_mode + ' graph')
            # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
            ax.set_yticklabels(input_files[index], fontsize=5)
            plt.grid()
            plt.savefig(outdir + 'Thr_graphcut_precisions_' + law_type + '_' + graph_mode + '_boxplot.png',
                        bbox_inches='tight')

    elif gradients_dir == 'gradients':

        input_files = os.listdir(hdf5_indir_grad)
        all_f_scores = []
        all_precisions = []
        all_recalls = []
        for gradients_input_file in input_files:
            with h5py.File(hdf5_indir_grad / gradients_input_file / 'gradients.h5', "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_file)

                gradient_vectors = gradients_file["perceptual_gradients"][:]
                gradient_shapes = gradients_file["gradient_shapes"][:]

                imgs_gradients = Parallel(n_jobs=num_cores)(
                    delayed(np.reshape)(gradients, (shape[0], shape[1])) for gradients, shape in
                    zip(gradient_vectors, gradient_shapes))

                outdir = '../outdir/' + \
                         num_imgs_dir + \
                         'slic_level_segmentation/' + \
                         (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                         'SimpleSum_all_imgs/' + \
                         'threshold_graphcut_' + law_type + '_dist_' + graph_mode + '_graph/' + \
                         gradients_input_file + '/'

                metrics_values = Parallel(n_jobs=num_cores)(
                    delayed(get_thr_graphcut_metrics)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                      graph_mode, law_type, cut_level, gradients_dir, outdir)
                    for im_file, img, regions_slic, graph_raw, perceptual_gradients in
                    zip(img_ids, images, superpixels, raw_graphs, imgs_gradients))

                outdir += 'results/'
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                metrics_values = np.array(metrics_values)
                recall = metrics_values[:, 0]
                all_recalls.append(recall)

                precision = metrics_values[:, 1]
                all_precisions.append(precision)

                f_score = hmean((precision, recall), axis=0)
                all_f_scores.append(f_score)

                plt.figure(dpi=180)
                plt.plot(np.arange(len(images)) + 1, recall, '-o', c='k', label='recall')
                plt.plot(np.arange(len(images)) + 1, precision, '-o', c='r', label='precision')
                plt.title('Thr graphcut P/R histogram')
                plt.xlabel(
                    'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                        recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                        precision.min(), precision.mean(), np.median(precision), precision.std()))
                plt.ylim(0, 1.05)
                plt.legend()
                plt.grid()
                plt.savefig(outdir + 'Thr_graphcut_PR_hist.png', bbox_inches='tight')

                plt.figure(dpi=180)
                sns.distplot(recall, color='black', label='recall')
                sns.distplot(precision, color='red', label='precision')
                plt.title('Thr graphcut P/R density histogram')

                plt.legend()
                plt.grid()
                plt.savefig(outdir + 'Thr_graphcut_PR_density_hist.png', bbox_inches='tight')

                plt.figure(dpi=180)
                ax = plt.gca()
                ax.boxplot(list([precision, recall]))
                ax.set_title('Thr graphcut P/R density box plot')
                ax.set_xticklabels(['precision', 'recall'])
                ax.set_xlabel('F-score: %.3f' % np.median(f_score))
                plt.grid()
                plt.savefig(outdir + 'Thr_graphcut_PR_boxplot.png', bbox_inches='tight')

                plt.close('all')

        outdir = '../outdir/' + \
                 num_imgs_dir + \
                 'slic_level_segmentation/' + \
                 (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/' + \
                 'SimpleSum_all_imgs/' + \
                 'threshold_graphcut_' + law_type + '_dist_' + graph_mode + '_graph/'

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        all_f_scores = np.array(all_f_scores)
        index = np.argsort(np.median(all_f_scores, axis=1))
        input_files = np.array(input_files)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_f_scores[index].T, vert=False)
        ax.set_title('Thr graphcut F scores: ' + law_type + ' dist, ' + graph_mode + ' graph')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'Thr_graphcut_Fscores_' + law_type + '_' + graph_mode + '_boxplot.png',
                    bbox_inches='tight')

        all_recalls = np.array(all_recalls)
        index = np.argsort(np.median(all_recalls, axis=1))
        input_files = np.array(input_files)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_recalls[index].T, vert=False)
        ax.set_title('Thr graphcut recall: ' + law_type + ' dist, ' + graph_mode + ' graph')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'Thr_graphcut_recalls_' + law_type + '_' + graph_mode + '_boxplot.png',
                    bbox_inches='tight')

        all_precisions = np.array(all_precisions)
        index = np.argsort(np.median(all_precisions, axis=1))
        input_files = np.array(input_files)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_precisions[index].T, vert=False)
        ax.set_title('Thr graphcut precision: ' + law_type + ' dist, ' + graph_mode + ' graph')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'Thr_graphcut_precisions_' + law_type + '_' + graph_mode + '_boxplot.png',
                    bbox_inches='tight')


if __name__ == '__main__':
    num_imgs = 7

    n_slic = 500 * 4

    # Graph function parameters
    graph_type = '8nn'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    gradients_dir = 'gradients'  # 'predicted_gradients'

    # Segmentation parameters
    graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
    law_type = 'gamma'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
    cut_level = 0.9  # set threshold at the 90% quantile level

    threshold_graphcut_segmentation(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, graph_mode,
                                    law_type, cut_level)
