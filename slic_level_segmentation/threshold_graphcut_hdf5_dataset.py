import sys
import h5py

from pathlib import Path
sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *


def get_thr_graphcut_metrics(im_file, img, regions_slic, graph_raw, perceptual_gradients, outdir):
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
    fig_title = graph_mode + ' Weighted Graph (' + graph_type + ')'
    img_name = '_weighted_' + graph_type
    colbar_lim = (min(weights), max(weights))
    show_and_save_imgraph(img, regions_slic, graph_weighted, fig_title, img_name, fontsize, save_fig, output_dir, file_name,
                          colbar_lim)

    # # Show one SLIC region and its neighbors
    # region = 109
    # show_and_save_some_regions(img, regions, region, rag, save_fig, outdir, file_name)

    # # Edges weight distribution
    # fig_title = 'Edges Weight Distribution'
    # img_name = '_weight_dist'
    # show_and_save_dist(weights, thresh, params, fig_title, img_name, fontsize, save_fig, outdir, file_name)

    # RAG after cut
    fig_title = 'RAG after cut'
    img_name = '_thr_graph_aftercut'
    colbar_lim = (min(weights), max(weights))
    show_and_save_imgraph(img, regions_slic, graph_aftercut, fig_title, img_name, fontsize, save_fig,
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

    return metrics_vals


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7
    gradients_dir = 'predicted_gradients'
    bsd_subset = 'all'

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_spix = hdf5_dir / 'complete' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / 'complete' / gradients_dir
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_spix = hdf5_dir / '7images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '7images' / gradients_dir
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_spix = hdf5_dir / '25images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '25images' / gradients_dir
        num_imgs_dir = '25images/'

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
    # Graph function parameters
    graph_type = '8nn'  # Choose: 'complete', 'knn', 'rag', 'keps' (k defines the number of neighbors or the radius)

    raw_graphs = Parallel(n_jobs=num_cores)(
        delayed(get_graph)(img, regions_slic, graph_type) for img, regions_slic in
        zip(images, superpixels))

    # Segmentation parameters
    method = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence
    graph_mode = 'mst'  # Choose: 'complete' to use whole graph or 'mst' to use Minimum Spanning Tree
    law_type = 'log'  # Choose 'log' for lognorm distribution or 'gamma' for gamma distribution
    cut_level = 0.9  # set threshold at the 90% quantile level

    if gradients_dir == 'gradients':

        input_files = os.listdir(hdf5_indir_grad)
        all_f_scores = []
        all_precisions = []
        all_recalls = []
        for gradients_input_file in input_files:
            with h5py.File(hdf5_indir_grad / gradients_input_file, "r+") as gradients_file:
                print('Reading Berkeley features data set')
                print('File name: ', gradients_input_file)
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
                         'slic_level_segmentation/' + \
                         num_imgs_dir + \
                         'threshold_graphcut/' + \
                         method + '_' + graph_type + '/' + \
                         bsd_subset + '_' + gradients_dir + '/' + \
                         gradients_input_file[:-3] + '/' + \
                         law_type + '_distribution/' + \
                         graph_mode + '_graph/'

                metrics_values = Parallel(n_jobs=num_cores)(
                    delayed(get_thr_graphcut_metrics)(im_file, img, regions_slic, graph_raw, perceptual_gradients, outdir)
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
                plt.xlabel('Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
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
                 'slic_level_segmentation/' + \
                 num_imgs_dir + \
                 'threshold_graphcut/' + \
                 method + '_' + graph_type + '/' + \
                 bsd_subset + '_' + gradients_dir + '/'

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
        plt.savefig(outdir + 'Thr_graphcut_Fscores_' + law_type + '_' + graph_mode + '_boxplot.png', bbox_inches='tight')

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
        plt.savefig(outdir + 'Thr_graphcut_recalls_' + law_type + '_' + graph_mode + '_boxplot.png', bbox_inches='tight')

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
        plt.savefig(outdir + 'Thr_graphcut_precisions_' + law_type + '_' + graph_mode + '_boxplot.png', bbox_inches='tight')

    if gradients_dir == 'predicted_gradients':

        model_input_dirs = os.listdir(hdf5_indir_grad)
        for mm, model_name in enumerate(model_input_dirs):
            input_files = os.listdir(hdf5_indir_grad / model_name)
            all_f_scores = []
            all_precisions = []
            all_recalls = []
            for gradients_input_file in input_files:
                with h5py.File(hdf5_indir_grad / model_name / gradients_input_file, "r+") as gradients_file:
                    print('Reading Berkeley features data set')
                    print('File name: ', gradients_input_file)
                    t0 = time.time()

                    imgs_gradients = np.array(gradients_file["/predicted_gradients"])

                    outdir = '../outdir/' + \
                             'slic_level_segmentation/' + \
                             num_imgs_dir + \
                             'threshold_graphcut/' + \
                             method + '_' + graph_type + '/' + \
                             gradients_dir + '/' + \
                             model_name + '/' + \
                             gradients_input_file[:-3] + '/' + \
                             law_type + '_distribution/' + \
                             graph_mode + '_graph/'

                    metrics_values = Parallel(n_jobs=num_cores)(
                        delayed(get_thr_graphcut_metrics)(im_file, img, regions_slic, graph_raw, perceptual_gradients,
                                                          outdir)
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
                     'slic_level_segmentation/' + \
                     num_imgs_dir + \
                     'threshold_graphcut/' + \
                     method + '_' + graph_type + '/' + \
                     gradients_dir + '/' + \
                     model_name + '/'

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



