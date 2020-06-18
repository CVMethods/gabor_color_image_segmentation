"""This utility converts a directory with files from the Berkeley Segmentation Dataset [BSD] (
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ into a petastorm dataset (Parquet format). The
script can run locally (use '--master=local[*]' command line argument), or submitted to a spark cluster. Schema
defined in examples.imagenet.schema.ImagenetSchema will be used. The schema NOTE: BSD needs to be
requested and downloaded separately by the user. """

import argparse
import glob
import json
import os
import pdb
import numpy as np

from scipy.stats import hmean

from skimage import io
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec

from BSD_metrics.groundtruth import *

BSDSchema = Unischema('BSDSchema', [
    UnischemaField('img_id', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('subdir', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('image', np.uint8, (None, None, 3), CompressedImageCodec('jpg'), False),
    UnischemaField('img_shape', np.int64, (None,), NdarrayCodec(), False),
    UnischemaField('ground_truth', np.uint16, (None, None, None), NdarrayCodec(), False),
    UnischemaField('num_seg', np.int64, (4,), NdarrayCodec(), False)
])


def get_num_segments(segments):
    n_labels = []
    for truth in segments:
        n_labels.append(len(np.unique(truth)))
    n_labels = np.array(n_labels)

    return np.array((n_labels.max(), n_labels.min(), int(n_labels.mean()), int(hmean(n_labels))))


def bsd_directory_to_petastorm_dataset(bsd_path, output_url, spark_master=None, parquet_files_count=100):
    """Converts a directory with bsd data into a petastorm dataset.
    Expected directory format is:
    >>> val/
    >>>    *.jpg
    >>> test/
    >>>    *.jpg
    >>> train/
    >>>    *.jpg
    :param bsd_path: a path to the directory containing ``*/`` subdirectories.
    :param output_url: the location where your dataset will be written to. Should be a url: either
      ``file://...`` or ``hdfs://...``
    :param spark_master: A master parameter used by spark session builder. Use default value (``None``) to use system
      environment configured spark cluster. Use ``local[*]`` to run on a local box.
    :return: ``None``
    """
    session_builder = SparkSession \
        .builder \
        .appName('BSD Features Dataset Creation') \
        .config('spark.executor.memory', '10g') \
        .config('spark.driver.memory', '10g')  # Increase the memory if running locally with high number of executors
    if spark_master:
        session_builder.master(spark_master)

    spark = session_builder.getOrCreate()
    sc = spark.sparkContext

    # list of [(img_id, 'subdir', path), ...]
    subdirectories = os.listdir(bsd_path)
    berkeley_dataset_list = []
    for subdir in subdirectories:
        imgs_path = bsd_path + subdir + "/"
        list_imgs = os.listdir(imgs_path)
        for file_name in list_imgs:
            groundtruth_segments = np.array(get_segment_from_filename(file_name[:-4]))
            n_segments = get_num_segments(groundtruth_segments)
            image = io.imread(imgs_path + file_name)
            img_shape = np.array(image.shape)
            berkeley_dataset_list.append((file_name[:-4], subdir, image, img_shape, groundtruth_segments,
                                          n_segments))

    ROWGROUP_SIZE_MB = 256
    with materialize_dataset(spark, output_url, BSDSchema, ROWGROUP_SIZE_MB):

        # rdd of [(img_id, 'subdir', image), ...]
        berkeley_dataset_rdd = sc.parallelize(berkeley_dataset_list) \
            .map(lambda id_image_list:
                 {BSDSchema.img_id.name: id_image_list[0],
                  BSDSchema.subdir.name: id_image_list[1],
                  BSDSchema.image.name: id_image_list[2],
                  BSDSchema.img_shape.name: id_image_list[3],
                  BSDSchema.ground_truth.name: id_image_list[4],
                  BSDSchema.num_seg.name: id_image_list[5]
                  })

        # Convert to pyspark.sql.Row
        sql_rows_rdd = berkeley_dataset_rdd.map(lambda r: dict_to_spark_row(BSDSchema, r))

        # Write out the result
        spark.createDataFrame(sql_rows_rdd, BSDSchema.as_spark_schema()) \
            .coalesce(parquet_files_count) \
            .write \
            .mode('overwrite') \
            .option('compression', 'none') \
            .parquet(output_url)  # , partitionBy='img_id'


if __name__ == '__main__':
    bsd_path = 'data/Berkeley/'
    output_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/complete/Berkeley_images'

    # bsd_path = 'data/myFavorite_BSDimages/'
    # output_url = 'file://' + os.getcwd() + '/data/petastorm_datasets/test/Berkeley_images'
    bsd_directory_to_petastorm_dataset(bsd_path, output_url)
