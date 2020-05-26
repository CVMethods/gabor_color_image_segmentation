Superpixels metrics
===============================

Computes evaluation metrics on the Berkeley Segmentation Dataset.


Data Organization
-----------------
|
|--data/     *# Folder containing the original images and the ground truth for the BSD dataset*


Scripts
-----------------
|
|--metrics.py       *# Implementation of a class handling the metrics calculation*
|--groundtruth.py   *# Implementation of Python methods to load the dataset*
|--script.py        *# Example script on how to use the code*


Requirements
---------------------------
| python3.x, scikit-image library, numpy


Description
---------------------------
Several metrics are considered in the file "metrics.py", namely:
- boundary recall
- boundary precision
- undersegmentation (Van den Bergh AND Neuber and Protzel formulae) 
- compactness
- density

For additional information, see the article:


Remarks
---------------------------
1. By default, in the current implementation, the boundary recall/precision are computed by using a square structuring element with size 5.
Keep in mind that in some articles, the structuring element might be different.
2. To compute the compactness metric, one has to compute the perimeter of each superpixels. Keep in mind that the definition of the perimeter might differ 
between implementations. Here, the perimeter is computed by counting the number of pixels in each superpixel that are either neighbors of pixels from 
another superpixel or located on the border of the image. A 4-neighborhood is considered.


