# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os, time, pdb

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
import scipy.cluster.vq as vq
from sklearn.preprocessing import StandardScaler

from skimage.io import imread
from skimage.segmentation import slic
from sklearn.decomposition import PCA

from BSD_metrics.groundtruth import *
from BSD_metrics.metrics import *
from myGaborFilter.myGaborFilter import *
from complexColor.color_transformations import *


if __name__ == '__main__':
    # img_path = "data/myFavorite_BSDimages/"
    img_path = "data/myFavorite_BSDimages/"
    outdir = 'data/outdir/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    names = sorted(os.listdir(img_path))

    # Generating Gabor filterbank
    min_period = 2.
    max_period = 25.
    fb = 1
    ab = 45
    c1 = 0.9
    c2 = 0.5
    stds = 3.0
    gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

    n_freq = len(frequencies)
    n_angles = len(angles)

    # # Visualization of filters
    fig1, axes1 = plt.subplots(n_freq, n_angles, dpi=120)
    ff = 0
    for ii, f_i in enumerate(frequencies):
        for jj, a_i in enumerate(angles):
            axes1[ii, jj].imshow(gabor_filters[ff][0].real, cmap='gray')
            axes1[ii, jj].tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                                      length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
            ff += 1
    axes1[n_freq - 1, np.int(np.ceil(n_angles / 2))].set_xlabel('Orientation   $\\theta_j $   $\\rightarrow$',
                                                                fontsize=10)
    axes1[np.int(np.ceil(n_freq / 2)), 0].set_ylabel('Frequency   $f_i$   $\\rightarrow$', fontsize=10)
    fig1.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45, wspace=0.45)
    fig1.suptitle('Gabor filterbank ', fontsize=10)

    lbls = [2, 4, 4, 2, 3, 3, 2]
    # lbls = [3, 3, 3, 3, 3, 3, 3]
    l = 0
    img_metrics = []
    for name in names:

        # Load the input image
        print("Processing image " + name[:-4])
        img_orig = imread(img_path + name)
        rows, cols, channels = img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]

        # plt.figure(dpi=180)
        # plt.imshow(img_orig)
        # plt.title('Input image', fontsize=10)
        # plt.axis('off')


        color_space = 'HS'
        lum, chrom_r, chrom_i = img2complex_colorspace(img_orig, color_space)

        ##################################  Luminance and chrominance normalization ##################################
        lum = linear_normalization(lum, 255., 0.)
        chrom_r = linear_normalization2(chrom_r)
        chrom_i = linear_normalization2(chrom_i)

        img_2ch = np.array((lum, chrom_r, chrom_i))
        img_2ch_norm = normalize_img(img_2ch, rows, cols)  # * (rows*cols)

        lum = img_2ch_norm[0]  # normalize_img(lum, rows, cols) #*np.sqrt(rows*cols)
        chrom_r = img_2ch_norm[1]  # normalize_img(chrom.real, rows, cols) #*np.sqrt(rows*cols)
        chrom_i = img_2ch_norm[2]  # normalize_img(chrom.imag, rows, cols) #*np.sqrt(rows*cols)

        # fig, axs = plt.subplots(1, 3, dpi=180)
        # axs[0].imshow(lum, cmap='gray')
        # axs[0].set_title('Luminance')
        # axs[1].imshow(chrom_r, cmap='gray')
        # axs[1].set_title('Chrominance (real part)')
        # axs[2].imshow(chrom_i, cmap='gray')
        # axs[2].set_title('Chrominance (imag part)')

        # print('Sum of the square values of the 2 channel image divided by the n° of pixels: ', np.sum(np.abs(img_2ch_norm) ** 2) / (rows * cols))

        ################################## Gabor filtering stage ##################################

        ############## Luminance ##############
        r_type = 'L2'  # 'real'
        gsmooth = True
        opn = True
        selem_size = 1

        g_responses_lum = applyGabor_filterbank(lum, gabor_filters, resp_type=r_type, smooth=gsmooth, morph_opening=opn, se_z=selem_size)

        ############## Chrominance real ##############

        g_responses_cr = applyGabor_filterbank(chrom_r, gabor_filters, resp_type=r_type, smooth=gsmooth, morph_opening=opn, se_z=selem_size)

        ############## Chrominance imag ##############

        g_responses_ci = applyGabor_filterbank(chrom_i, gabor_filters, resp_type=r_type, smooth=gsmooth, morph_opening=opn, se_z=selem_size)

        ################################## Gabor responses normalization ##################################

        g_responses = np.array([g_responses_lum, g_responses_cr, g_responses_ci])
        g_responses_norm = normalize_img(g_responses, rows, cols)  # * (rows*cols) # g_responses / np.sum(np.abs(np.array([g_responses_lum, g_responses_cr, g_responses_ci]))**2)

        g_responses_lum = g_responses_norm[0]
        g_responses_cr = g_responses_norm[1]
        g_responses_ci = g_responses_norm[2]

        vmax = g_responses_norm.max()
        vmin = g_responses_norm.min()

        # print('[Min, Max] values among lum, chrom responses after normalization: ', [vmin, vmax])

        ############## Luminance ##############
        # fig, axes = plt.subplots(n_freq, n_angles, dpi=180)
        # ff = 0
        # for ii, f_i in enumerate(frequencies):
        #     for jj, a_i in enumerate(angles):
        #         axes[ii, jj].imshow(g_responses_lum[ff], cmap='gray', vmin=vmin, vmax=vmax)  #
        #         axes[ii, jj].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False,
        #                                  labelleft=False)
        #         ff += 1
        # axes[n_freq - 1, np.int(np.ceil(n_angles / 2))].set_xlabel('Orientation   $\\theta_j $   $\\rightarrow$',
        #                                                            fontsize=10)
        # axes[np.int(np.ceil(n_freq / 2)), 0].set_ylabel('Frequency   $f_i$   $\\rightarrow$', fontsize=10)
        # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        # fig.suptitle('Gabor energy of luminance channel', fontsize=10)

        ############## Chrominance real ##############
        # fig, axes = plt.subplots(n_freq, n_angles, dpi=180)
        # ff = 0
        # for ii, f_i in enumerate(frequencies):
        #     for jj, a_i in enumerate(angles):
        #         axes[ii, jj].imshow(g_responses_cr[ff], cmap='gray', vmin=vmin, vmax=vmax)  #
        #         axes[ii, jj].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False,
        #                                  labelleft=False)
        #         ff += 1
        # axes[n_freq - 1, np.int(np.ceil(n_angles / 2))].set_xlabel('Orientation   $\\theta_j $   $\\rightarrow$',
        #                                                            fontsize=10)
        # axes[np.int(np.ceil(n_freq / 2)), 0].set_ylabel('Frequency   $f_i$   $\\rightarrow$', fontsize=10)
        # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        # fig.suptitle('Gabor energy of chrominance (real part) channel', fontsize=10)

        ############## Chrominance imag ##############
        # fig, axes = plt.subplots(n_freq, n_angles, dpi=180)
        # ff = 0
        # for ii, f_i in enumerate(frequencies):
        #     for jj, a_i in enumerate(angles):
        #         axes[ii, jj].imshow(g_responses_ci[ff], cmap='gray', vmin=vmin, vmax=vmax)  #
        #         axes[ii, jj].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False,
        #                                  labelleft=False)
        #         ff += 1
        # axes[n_freq - 1, np.int(np.ceil(n_angles / 2))].set_xlabel('Orientation   $\\theta_j $   $\\rightarrow$',
        #                                                            fontsize=10)
        # axes[np.int(np.ceil(n_freq / 2)), 0].set_ylabel('Frequency   $f_i$   $\\rightarrow$', fontsize=10)
        # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        # fig.suptitle('Gabor energy of chrominance (imag part) channel', fontsize=10)

        # print('Sum of the square values of the Gabor resp divided by the n° of pixels: ', np.sum(g_responses_norm ** 2) / (rows * cols))

        ################################## Kmeans N°1 ##################################

        X = []
        for ff in range(g_responses_lum.shape[0]):
            X.append(reshape4clustering(g_responses_lum[ff], rows, cols))
            X.append(reshape4clustering(g_responses_cr[ff], rows, cols))
            X.append(reshape4clustering(g_responses_ci[ff], rows, cols))

        X = np.array(X).T
        # X = vq.whiten(X)
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=3)
        reduced_X = pca.fit_transform(X)
        # pixels = np.arange(rows * cols)
        # nodes = pixels.reshape((rows, cols))
        # yy, xx = np.where(nodes >= 0)
        # X = np.column_stack((X, yy, xx))

        nc = 3  # lbls[l]
        # l += 1
        clustering = KMeans(n_clusters=nc, random_state=0, n_jobs=-1).fit(reduced_X)
        # # clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=4, eigen_solver='amg', n_jobs=-1).fit(reduced_X)
        # clustering = DBSCAN(eps=0.9, n_jobs=-1).fit(reduced_X)
        # clustering = AgglomerativeClustering(eps=0.9, n_jobs=-1).fit(reduced_X)

        kmeans_labels = clustering.labels_.reshape((rows, cols))

        fig = plt.figure(dpi=180)
        ax = plt.gca()
        im = ax.imshow(kmeans_labels, cmap=plt.cm.get_cmap('Set1', nc))
        ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                       length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
        ax.set_title('KMeans labels, k=%d' % nc, fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax, ticks=(np.arange(nc) + 0.5) * (nc - 1) / nc)
        cb.set_label(label='$labels$', fontsize=10)
        cb.ax.tick_params(labelsize=6)
        cb.ax.set_yticklabels([r'${{{}}}$'.format(val) for val in range(nc)])
        plt.savefig(outdir + name[:-4]+'_segm.png')

        ## Performs the segmentation
        # labels = slic(img, n_segments=300, compactness=10.0)

        # Load the ground truth
        segments = get_segment_from_filename(name[:-4])
        # Evaluate metrics
        m = metrics(img_orig, kmeans_labels, segments)
        m.set_metrics()
        # m.display_metrics()

        metrics_values = list(m.get_metrics().values())
        # dict_metrics = m.get_metrics()
        # dict_metrics.update({'image_name': name[:-4]})
        img_metrics.append(metrics_values)


    img_metrics = np.array(img_metrics)
    recall_avg = np.mean(img_metrics[:, 1])
    precision_avg = np.mean(img_metrics[:, 2])
    underseg_avg = np.mean(img_metrics[:, 3])
    undersegNP_avg = np.mean(img_metrics[:, 4])
    compactness_avg = np.mean(img_metrics[:, 5])
    density_avg = np.mean(img_metrics[:, 6])

    recall_std = np.std(img_metrics[:, 1])
    precision_std = np.std(img_metrics[:, 2])
    underseg_std = np.std(img_metrics[:, 3])
    undersegNP_std = np.std(img_metrics[:, 4])
    compactness_std = np.std(img_metrics[:, 5])
    density_std = np.std(img_metrics[:, 6])

    print(" Avg Recall: " + str(recall_avg) + "\n",
          " Avg Precision: " + str(precision_avg) + "\n",
          " Avg Undersegmentation (Bergh): " + str(underseg_avg) + "\n",
          " Avg Undersegmentation (NP): " + str(undersegNP_avg) + "\n",
          " Avg Compactness: " + str(compactness_avg) + "\n",
          " Avg Density: " + str(density_avg) + "\n")

    print(" Std Recall: " + str(recall_std) + "\n",
          " Std Precision: " + str(precision_std) + "\n",
          " Std Undersegmentation (Bergh): " + str(underseg_std) + "\n",
          " Std Undersegmentation (NP): " + str(undersegNP_std) + "\n",
          " Std Compactness: " + str(compactness_std) + "\n",
          " Std Density: " + str(density_std) + "\n")

    # plt.show()
