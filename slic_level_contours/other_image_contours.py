import sys
import h5py

from pathlib import Path

sys.path.append('../')
from source.metrics import *
from source.groundtruth import *
from source.graph_operations import *
from source.plot_save_figures import *
from source.color_seg_methods import *

from skimage.transform import resize
from skimage.color import rgb2gray

def get_img_contours(im_file, img, detector, outdir):
    print('##############################', im_file, '##############################')
    if method == 'canny':
        img = np.uint8(rgb2gray(img)*255)
        img_grad = cv2.Canny(img, 100, 200)

    else:
        img = resize(img, (int(img.shape[0]*1), int(img.shape[1]*1)), mode="reflect")
        img_grad = detector.detectEdges(img.astype(np.float32))
        # img_grad = resize(img_grad, (int(img.shape[0]), int(img.shape[1])), mode="reflect")
        img_grad = np.uint8(img_grad*255)

    ##############################################################################
    '''Visualization Section: show and/or save images'''
    img = Image.fromarray(img_grad)
    img.save(outdir + im_file + '.png')
    # plt.imsave(outdir + im_file + '.png', img_grad, cmap='gray')

    return img_grad


def generate_imgcontours_with_sed(num_imgs, method):
    num_cores = -1
    hdf5_indir_im = Path('../../data/hdf5_datasets/' + str(num_imgs) + 'images/' + 'images')
    models_indir = '../../data/models/'
    num_imgs_dir = str(num_imgs) + 'images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = np.array(images_file["/images"])
    img_shapes = np.array(images_file["/image_shapes"])
    img_ids = np.array(images_file["/image_ids"])
    img_subdirs = np.array(images_file["/image_subdirs"])

    images = np.array(Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes)))

    test_indices = []
    for ii in range(len(images)):
        if img_subdirs[ii] == 'test':
            test_indices.append(ii)

    img_ids = img_ids[test_indices]
    images = images[test_indices]

    outdir = '../outdir/' + \
             num_imgs_dir + \
             'image_contours/oOther_methods/no_Gabor/' + \
             method + '/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if method == 'structured_forest':
        detector = cv2.ximgproc.createStructuredEdgeDetection(models_indir + 'opencv_sed_model.yml.gz')

        img_contours = Parallel(n_jobs=1)(
                                delayed(get_img_contours)(im_file, img, detector, outdir)
                                for im_file, img in zip(img_ids, images))

    if method == 'canny':
        detector = method
        img_contours = Parallel(n_jobs=num_imgs)(
                                delayed(get_img_contours)(im_file, img, detector, outdir)
                                for im_file, img in zip(img_ids, images))

if __name__ == '__main__':
    num_imgs = 500
    method = 'structured_forest'  #'canny' #

    generate_imgcontours_with_sed(num_imgs, method)
