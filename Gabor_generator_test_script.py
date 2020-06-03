import matplotlib.pyplot as plt
from myGaborFilter.myGaborFilter import makeGabor_filter, makeGabor_filterbank
import numpy as np
from skimage.filters import gabor_kernel as gabor_skimage

gabor = makeGabor_filter(frequency=1/3, angle=0, freq_bandwidth=1, freq_crossing_point=0.9, angle_bandwidth=30, ang_crossing_point=0.5)

plt.figure()
plt.imshow(gabor[0].real, cmap='gray')
# gabor_sk = gabor_skimage(frequency=1/4, theta=0)
# plt.figure()
# plt.imshow(gabor_sk.real, cmap='gray')
# plt.show()

# Generating Gabor filterbank
min_period = 2.
max_period = 35.
fb = 1
ab = 45
c1 = 0.9
c2 = 0.5
stds = 3.5
gabor_filters, frequencies, angles = makeGabor_filterbank(min_period, max_period, fb, ab, c1, c2, stds)

n_freq = len(frequencies)
n_angles = len(angles)

# Visualization of filters
fig1, axes1 = plt.subplots(n_freq, n_angles, dpi=120)
ff = 0
for ii, f_i in enumerate(frequencies):
    for jj, a_i in enumerate(angles):
        axes1[ii, jj].imshow(gabor_filters[ff][0].real, cmap='gray')
        axes1[ii, jj].tick_params(axis='both', which='both', labelsize=7, pad=0.1, length=2)#, bottom=False, left=False, labelbottom=False, labelleft=False
        ff += 1
axes1[n_freq-1, np.int(np.ceil(n_angles/2))].set_xlabel('Orientation   $\\theta_j $   $\\rightarrow$', fontsize=10)
axes1[np.int(np.ceil(n_freq/2)), 0].set_ylabel('Frequency   $f_i$   $\\rightarrow$', fontsize=10)
fig1.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45, wspace=0.45)
fig1.suptitle('Gabor filterbank ', fontsize=10)
# figcaption('Gabor filter bank', label="fig:gabor_filterbank")
plt.show()