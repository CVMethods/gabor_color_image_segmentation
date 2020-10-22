import sys
import pandas as pd

from pathlib import Path

sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *

def process_contours_fscores(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset):
    num_cores = -1

    num_imgs_dir = str(num_imgs) + 'images/'
    contours_indir = '../outdir/' + num_imgs_dir + 'image_contours/' + (str(n_slic) + '_slic_' + graph_type + '_' + similarity_measure) + '/'

    model_input_dirs = sorted(os.listdir(contours_indir))
    for model_name in model_input_dirs:
        input_directories = sorted([f for f in os.listdir(contours_indir + model_name) if not f.endswith('png')])

        all_f_scores = []
        all_precisions = []
        all_recalls = []

        ods = []
        ois = []
        ap = []

        for gabor_config in input_directories:
            scores_indir = contours_indir + model_name + '/' + gabor_config + '/'
            outdir = scores_indir

            optimal_scores_dataframe = pd.read_csv(scores_indir + 'eval_bdry.txt', delimiter='\s+', header=None)
            optimal_scores_dataframe.columns = ["best Thr", "best recall", "best precision", "best fscore", "recall max", "precision max", "fscore max", "average precision"]
            optimal_scores_array = optimal_scores_dataframe.to_numpy().flatten()

            ods.append(optimal_scores_array[3])
            ois.append(optimal_scores_array[6])
            ap.append(optimal_scores_array[7])

            scores_dataframe = pd.read_csv(scores_indir + 'eval_bdry_img.txt', delimiter='\s+', header=None)
            scores_dataframe.columns = ["num img", "threshold", "recall", "precision", "fscore"]
            scores_array = scores_dataframe.to_numpy()

            recall = scores_array[:, 2]
            all_recalls.append(recall)

            precision = scores_array[:, 3]
            all_precisions.append(precision)

            f_score = scores_array[:, 4]
            all_f_scores.append(f_score)

            plt.figure(dpi=180)
            plt.plot(np.arange(len(scores_array)) + 1, recall, '-o', c='k', label='recall')
            plt.plot(np.arange(len(scores_array)) + 1, precision, '-o', c='r', label='precision')
            plt.title('Thr graphcut P/R histogram')
            plt.xlabel(
                'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, Pmed: %.3f, Pstd: %.3f ' % (
                    recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(), precision.max(),
                    precision.min(), precision.mean(), np.median(precision), precision.std()))
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'PR_hist.png', bbox_inches='tight')

            plt.figure(dpi=180)
            sns.distplot(recall, color='black', label='recall')
            sns.distplot(precision, color='red', label='precision')
            plt.title('Thr graphcut P/R density histogram')

            plt.legend()
            plt.grid()
            plt.savefig(outdir + 'PR_density_hist.png', bbox_inches='tight')

            plt.figure(dpi=180)
            ax = plt.gca()
            ax.boxplot(list([precision, recall]))
            ax.set_title('Thr graphcut P/R density box plot')
            ax.set_xticklabels(['precision', 'recall'])
            ax.set_xlabel('F-score: %.3f' % np.median(f_score))
            plt.grid()
            plt.savefig(outdir + 'PR_boxplot.png', bbox_inches='tight')

            plt.close('all')

        outdir = contours_indir + model_name + '/'
        all_f_scores = np.array(all_f_scores)
        index = np.argsort(np.median(all_f_scores, axis=1))
        input_files = np.array(input_directories)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_f_scores[index].T, vert=False)
        ax.set_title('All F scores')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'fscores_boxplot.png', bbox_inches='tight')

        all_recalls = np.array(all_recalls)
        index = np.argsort(np.median(all_recalls, axis=1))
        input_files = np.array(input_files)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_recalls[index].T, vert=False)
        ax.set_title('All recall')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'recalls_boxplot.png', bbox_inches='tight')

        all_precisions = np.array(all_precisions)
        index = np.argsort(np.median(all_precisions, axis=1))
        input_files = np.array(input_files)

        plt.figure(dpi=180)
        ax = plt.gca()
        ax.boxplot(all_precisions[index].T, vert=False)
        ax.set_title('All precision')
        # ax.legend(input_files, fontsize=5, loc='best', bbox_to_anchor=(1, 1))
        ax.set_yticklabels(input_files[index], fontsize=5)
        plt.grid()
        plt.savefig(outdir + 'precisions_boxplot.png', bbox_inches='tight')
        # print('Reading Berkeley image data set')

        ods = np.array(ods)
        ois = np.array(ois)
        ap = np.array(ap)

        y = np.arange(len(input_files))  # the label locations
        width = 0.2  # the width of the bars
        index = np.argsort(ods)

        fig, ax = plt.subplots()
        rects1 = ax.barh(y - width,  ods[index], width, label='ODS')
        rects2 = ax.barh(y, ois[index], width, label='OIS')
        rects3 = ax.barh(y + width, ap[index], width, label='AP')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Gabor configurations')
        ax.set_xlabel('Score')
        ax.set_title('Scores by group and gender')
        ax.set_yticks(y)
        ax.set_yticklabels(input_files[index])
        ax.legend()
        plt.grid()

        plt.savefig(outdir + 'optimal_scores_barchart.png', bbox_inches='tight')






if __name__ == '__main__':
    num_imgs = 7

    base = 500

    # Graph function parameters
    graph_type = 'rag'  # Choose: 'complete', 'knn', 'rag', 'eps'

    # Distance parameter
    similarity_measure = 'OT'  # Choose: 'OT' for Earth Movers Distance or 'KL' for Kullback-Leiber divergence

    gradients_dir = 'predicted_gradients'  # 'predicted_gradients'
    bsd_subset = 'all'

    for ns in [3, 5, 7, 9, 11]:
        n_slic = base * ns
        process_contours_fscores(num_imgs, n_slic, graph_type, similarity_measure, gradients_dir, bsd_subset)
