# -*- coding: utf-8 -*-
# !/usr/bin/env python

import time, warnings, pdb, os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyarrow.parquet import ParquetDataset

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
    UnischemaField('subdir', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('image', np.uint8, (None, None, 3), CompressedImageCodec('jpg'), False),
    UnischemaField('img_shape', np.int64, (None,), NdarrayCodec(), False),
    UnischemaField('ground_truth', np.uint16, (None, None, None), NdarrayCodec(), False),
    UnischemaField('num_seg', np.int64, (4,), NdarrayCodec(), False),
    UnischemaField('complex_image', np.float32, (3, None, None), NdarrayCodec(), False),
    UnischemaField('gabor_features', np.float_, (None, None), NdarrayCodec(), False)
])

def get_gabor_features(img_complex, gabor_filters, r_type, gsmooth, opn, selem_size):

    channels, rows, cols = img_complex.shape
    filter_responses = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(applyGabor_filterbank)(img_channel, gabor_filters, resp_type=r_type, smooth=gsmooth,
                                       morph_opening=opn, se_z=selem_size) for img_channel in img_complex))

    g_responses_norm = normalize_img(filter_responses, rows, cols)
    # print(np.sum(g_responses_norm**2) / (rows*cols))

    features = []
    for ff in range(g_responses_norm.shape[1]):
        features.append(reshape4clustering(g_responses_norm[0][ff], rows, cols))
        features.append(reshape4clustering(g_responses_norm[1][ff], rows, cols))
        features.append(reshape4clustering(g_responses_norm[2][ff], rows, cols))

    features = np.array(features).T

    return features


def row_generator(im_id, subdir, img,  shape, truth, nsegs, img_complex, features):
    return {'img_id': im_id,
            'subdir': subdir,
            'image': img,
            'img_shape': shape,
            'ground_truth': truth,
            'num_seg': nsegs,
            'complex_image': img_complex,
            'gabor_features': features
            }


def bsd_features_to_petastorm_dataset(features_list, output_url, spark_master=None, parquet_files_count=100):

    session_builder = SparkSession \
        .builder \
        .appName('/BSD Features Dataset Creation') #\
        # .config('spark.executor.memory', '400g') \
        # .config('spark.driver.memory', '400g')  # Increase the memory if running locally with high number of executors
    if spark_master:
        session_builder.master(spark_master)

    spark = session_builder.getOrCreate()
    sc = spark.sparkContext
    print('\nSaving Gabor features in petastorm data set')
    ROWGROUP_SIZE_MB = 256
    with materialize_dataset(spark, output_url, BSDFeaturesSchema, ROWGROUP_SIZE_MB):

        # rdd of [(img_id, 'subdir', image), ...] & Convert to pyspark.sql.Row
        sql_rows_rdd = sc.parallelize(features_list, numSlices=min(len(features_list) / 10 + 1, 10000)).map(lambda r: dict_to_spark_row(BSDFeaturesSchema, r))  #

        # Write out the result
        spark.createDataFrame(sql_rows_rdd, BSDFeaturesSchema.as_spark_schema()) \
            .coalesce(parquet_files_count) \
            .write \
            .mode('overwrite') \
            .option('compression', 'none') \
            .parquet(output_url)   # , partitionBy='img_id'


if __name__ == '__main__':

    # dataset_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/test/Berkeley_images'
    # output_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/test/Berkeley_GaborFeatures'

    dataset_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/complete/Berkeley_images'
    output_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/complete/Berkeley_GaborFeatures'


    # Generating Gabor filterbank
    min_period = 2.
    max_period = 35.
    fb = 0.7  # 1 #
    ab = 30  # 45 #
    c1 = 0.9
    c2 = 0.7
    stds = 3.5
    print('Generating Gabor filterbank')
    gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

    n_freq = len(frequencies)
    n_angles = len(angles)
    output_url += '_%dfreq_%dang' % (n_freq, n_angles)
    color_space = 'HS'

    r_type = 'L2'  # 'real'
    gsmooth = True
    opn = True
    selem_size = 1
    num_cores = -1

    # dataset_schema_headers = list(ParquetDataset(dataset_path).read_pandas().to_pandas())
    # for header in dataset_schema_headers:
    #     exec("%s = []" % header)

    indices = []
    img_ids = []
    subdirs = []
    images = []
    img_shapes = []
    ground_truths = []
    num_segments = []

    print('Reading Berkeley image data set')
    with make_reader(dataset_url) as reader:
        for sample in reader:
            img_ids.append(sample.img_id.decode('UTF-8'))
            subdirs.append(sample.subdir.decode('UTF-8'))
            images.append(sample.image)
            img_shapes.append(sample.img_shape)
            ground_truths.append(sample.ground_truth)
            num_segments.append(sample.num_seg)

    # ## Parallel computation of Gabor features
    print('Computing Gabor features:')
    t0 = time.time()
    twoChannel_imgs = Parallel(n_jobs=num_cores)(
        delayed(img2complex_normalized_colorspace)(img, shape, color_space) for img, shape in zip(images, img_shapes))

    gabor_features = Parallel(n_jobs=num_cores, prefer='processes')(
        delayed(get_gabor_features)(img, gabor_filters, r_type, gsmooth, opn, selem_size) for img in twoChannel_imgs)
    t1 = time.time()
    print('Features computing time (using Parallel joblib): %.2fs' % (t1 - t0))

    iterator = zip(img_ids, subdirs, images, img_shapes, ground_truths, num_segments, twoChannel_imgs, gabor_features)
    bsd_features_list = Parallel(n_jobs=num_cores)(delayed(row_generator)(im_id, subdir, img,  shape, truth, nsegs, img_complex, features) for im_id, subdir, img,  shape, truth, nsegs, img_complex, features in iterator)

    bsd_features_to_petastorm_dataset(bsd_features_list, output_url)
    print('Berkeley Gabor features data set DONE!')
    #     # dataset_values_list = Parallel(n_jobs=num_cores)(delayed(petastorm2list)(sample, dataset_schema_headers) for sample in reader)
    # # ## Parallel computation of Gabor features
    # t0 = time.time()
    # with make_reader(dataset_url) as reader:
    #
    #     twoChannel_imgs = Parallel(n_jobs=num_cores)(
    #         delayed(img2complex_normalized_colorspace)(sample.image, sample.img_shape, color_space) for sample in reader)
    #
    # gabor_features = Parallel(n_jobs=num_cores, prefer='processes')(
    #     delayed(get_gabor_features)(img, gabor_filters, r_type, gsmooth, opn, selem_size) for img in twoChannel_imgs)
    # t1 = time.time()
    # print('Computing time using Parallel joblib: %.2fs' % (t1 - t0))
    # # list of [(img_id, 'complex_img', 'features', 'num_clusters'), ...]
    # with make_reader(dataset_url) as reader:
    #     bsd_features_list = Parallel(n_jobs=num_cores)(
    #         delayed(row_generator)(sample.img_id.decode('UTF-8'), twoChannel_imgs[ii], gabor_features[ii], sample.mean_max_min_nseg)
    #         for ii, sample in enumerate(reader))
    #
    #



    # img = imread(img_path + img_names[ii])
    # # xx, yy = img.shape
    # # img = img.resize((xx // ss, yy // ss))
    # # img = np.array(img)[:, :, 0:3]
    # yy, xx, zz = img.shape
    # X, y, img_size = img.reshape((xx * yy, zz)), None, (yy, xx)
    # datasets.append(((X, y, clusters[ii], img_size), {}))
    # # ## Simple for loop computation of Gabor features
    # t0 = time.time()
    # with make_reader(dataset_url) as reader:
    #     twoChannel_imgs_v2 = []
    #     gabor_features_v2 = []
    #     for sample in reader:
    #         img_2ch_norm = img2complex_normalized_colorspace(sample.image, sample.img_shape, color_space)
    #
    #         twoChannel_imgs_v2.append(img_2ch_norm)
    #         rows, cols, channels = sample.img_shape
    #         lum = img_2ch_norm[0]  # normalize_img(lum, rows, cols) #*np.sqrt(rows*cols)
    #         chrom_r = img_2ch_norm[1]  # normalize_img(chrom.real, rows, cols) #*np.sqrt(rows*cols)
    #         chrom_i = img_2ch_norm[2]  # normalize_img(chrom.imag, rows, cols) #*np.sqrt(rows*cols)
    #
    #
    #
    #         ################################## Gabor filtering stage ##################################
    #
    #         ############## Luminance ##############
    #
    #         g_responses_lum = applyGabor_filterbank(lum, gabor_filters, resp_type=r_type, smooth=gsmooth,
    #                                                 morph_opening=opn, se_z=selem_size)
    #
    #         ############## Chrominance real ##############
    #
    #         g_responses_cr = applyGabor_filterbank(chrom_r, gabor_filters, resp_type=r_type, smooth=gsmooth,
    #                                                morph_opening=opn, se_z=selem_size)
    #
    #         ############## Chrominance imag ##############
    #
    #         g_responses_ci = applyGabor_filterbank(chrom_i, gabor_filters, resp_type=r_type, smooth=gsmooth,
    #                                                morph_opening=opn, se_z=selem_size)
    #
    #         ################################## Gabor responses normalization ##################################
    #
    #         g_responses = np.array([g_responses_lum, g_responses_cr, g_responses_ci])
    #         g_responses_norm = normalize_img(g_responses, rows, cols)  # * (rows*cols) # g_responses / np.sum(np.abs(np.array([g_responses_lum, g_responses_cr, g_responses_ci]))**2)
    #
    #         g_responses_lum = g_responses_norm[0]
    #         g_responses_cr = g_responses_norm[1]
    #         g_responses_ci = g_responses_norm[2]
    #
    #         X = []
    #         for ff in range(g_responses_lum.shape[0]):
    #             X.append(reshape4clustering(g_responses_lum[ff], rows, cols))
    #             X.append(reshape4clustering(g_responses_cr[ff], rows, cols))
    #             X.append(reshape4clustering(g_responses_ci[ff], rows, cols))
    #
    #         X = np.array(X).T
    #         # # normalize dataset for easier parameter selection
    #         # X = StandardScaler().fit_transform(X)
    #         # X = vq.whiten(X)
    #
    #         gabor_features_v2.append(X)
    # t1 = time.time()
    # print('Computing time using for loop: %.2fs' % (t1 - t0))
