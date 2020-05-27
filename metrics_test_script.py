# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os, time, pdb

from math import *
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.segmentation import slic

from BSD_metrics.groundtruth import *
from BSD_metrics.metrics import *

# ---------------------------------
# when executed, run the main script
# ---------------------------------
if __name__ == '__main__':
    img_path = 'data/Berkeley/train/'
    names = os.listdir(img_path)
    for name in names[:5]:

        # Load the input image
        img = imread(img_path + name)

        print("Processing image " + name[:-4])

        # Performs the segmentation
        labels = slic(img, n_segments=300, compactness=10.0)

        # Load the ground truth
        segments = get_segment_from_filename(name[:-4])   

        # Evaluate metrics
        m = metrics(img, labels, segments)
        m.set_metrics()
        m.display_metrics()




