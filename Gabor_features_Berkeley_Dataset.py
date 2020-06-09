# -*- coding: utf-8 -*-
# !/usr/bin/env python

import time, warnings, pdb, os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
import scipy.cluster.vq as vq
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.segmentation import slic
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm import make_reader

from BSD_metrics.groundtruth import *
from BSD_metrics.metrics import *
from Gabor_analysis.myGaborFunctions import *
from complexColor.color_transformations import *

BSDFeaturesSchema = Unischema('BSDFeaturesSchema', [
    UnischemaField('img_id', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('complex_image', np.complex_, (None, None, 3), NdarrayCodec(), False),
    UnischemaField('gabor_feature_array', np.float_, (None, None, 3), NdarrayCodec(), False),
])


def img2_twochannel_colr_complex(img, color_space='HS'):

    rows, cols, channels = img.shape
    lum, chrom_r, chrom_i = img2complex_colorspace(img, color_space)

    ##################################  Luminance and chrominance normalization ##################################
    lum = linear_normalization(lum, 255., 0.)
    chrom_r = linear_normalization2(chrom_r)
    chrom_i = linear_normalization2(chrom_i)

    img_2ch = np.array((lum, chrom_r, chrom_i))
    img_2ch_norm = normalize_img(img_2ch, rows, cols)

    return img_2ch_norm

if __name__ == '__main__':
    dataset_url = 'file://' + os.getcwd() + '/data/Berkeley_petastorm_dataset_test'
    outdir = 'data/outdir/features/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)


    # Generating Gabor filterbank
    min_period = 2.
    max_period = 35.
    fb = 0.7
    ab = 30
    c1 = 0.9
    c2 = 0.7
    stds = 3.5
    gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

    n_freq = len(frequencies)
    n_angles = len(angles)
    with make_reader(dataset_url) as reader:
        pdb.set_trace()
        # Pure python
        ii = 0
        for sample in reader:
            print(sample.img_id, sample.subdir)
            ii += 1
            print(ii)
            rows, cols, channels = sample.image.shape
            # plt.figure(dpi=180)
            # plt.imshow(img_orig)
            # plt.title('Input image', fontsize=10)
            # plt.axis('off')

            color_space = 'HS'
            lum, chrom_r, chrom_i = img2complex_colorspace(sample.image, color_space)

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

            g_responses_lum = applyGabor_filterbank(lum, gabor_filters, resp_type=r_type, smooth=gsmooth, morph_opening=opn,
                                                    se_z=selem_size)

            ############## Chrominance real ##############

            g_responses_cr = applyGabor_filterbank(chrom_r, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                                   morph_opening=opn, se_z=selem_size)

            ############## Chrominance imag ##############

            g_responses_ci = applyGabor_filterbank(chrom_i, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                                   morph_opening=opn, se_z=selem_size)

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
            print(X.shape)
            # # normalize dataset for easier parameter selection
            # X = StandardScaler().fit_transform(X)
            # X = vq.whiten(X)
        pdb.set_trace()


