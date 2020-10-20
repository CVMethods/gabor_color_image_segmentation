import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from joblib import Parallel, delayed
from scipy.cluster import vq
from scipy.stats import hmean
from sklearn.decomposition import PCA
from skimage import segmentation
from sklearn import cluster

sys.path.append('../')
from source.groundtruth import *
from source.metrics import *
from source.computation_support import *


def clustering_segmentation(i_dataset, dataset, algo_params, num_clusters, algo_name):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    img_id, img, X, y, n_clusters, img_size = dataset

    print('dataset image:', img_id)

    # img_size = img_size[:-1]
    rows, cols, channels = img_size

    if num_clusters == 'max':
        params['n_clusters'] = int(n_clusters[0])
    elif num_clusters == 'min':
        params['n_clusters'] = int(n_clusters[1])
    elif num_clusters == 'mean':
        params['n_clusters'] = int(n_clusters[2])
    elif num_clusters == 'hmean':
        params['n_clusters'] = int(n_clusters[3])

    # print(params['n_clusters'], num_clusters, int(n_clusters[0]))

    # Add pixel's position to features to include locality
    pixels = np.arange(rows * cols)
    nodes = pixels.reshape((rows, cols))
    yy, xx = np.where(nodes >= 0)
    X = np.column_stack((X, yy, xx))

    # # normalize dataset for easier parameter selection
    # X = StandardScaler().fit_transform(X)
    X = vq.whiten(X)

    # Reduce data dimensionality (if needed for faster clustering computation)
    pca = PCA(n_components=4)
    X = pca.fit_transform(X)

    # Create cluster object
    algorithm = cluster.KMeans(params['n_clusters'], random_state=0, n_jobs=params['n_jobs'])

    t0 = time.time()
    algorithm.fit(X)
    t1 = time.time()

    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)

    nc = len(np.unique(y_pred))
    y_pred = y_pred.reshape((rows, cols))

    # Evaluate metrics
    m = metrics(None, y_pred, y)
    m.set_metrics()
    # m.display_metrics()

    metrics_values = m.get_metrics()

    plt.figure(dpi=180)
    out = color.label2rgb(y_pred, img, kind='avg')
    out = segmentation.mark_boundaries(out, y_pred, color=(0, 0, 0), mode='thick')
    ax = plt.gca()
    ax.imshow(out)
    ax.tick_params(axis='both', which='both', labelsize=7, pad=0.1,
                   length=2)  # , bottom=False, left=False, labelbottom=False, labelleft=False
    ax.set_title(algo_name + ' k=%d' % nc, fontsize=10)
    ax.set_xlabel(('Recall: %.3f, Precision: %.3f, Time: %.2fs' % (
        metrics_values['recall'], metrics_values['precision'], (t1 - t0))).lstrip('0'), fontsize=10)
    plt.savefig(outdir + '%02d' % i_dataset + '_' + img_id + '_' + algo_name + '_' + num_clusters + '_segm.png')
    plt.cla()
    plt.clf()
    plt.close()
    
    return metrics_values  # y_pred


def save_plot_metrics(ids, metrics, algo_name):
    algorithm_metrics = np.array(metrics)

    np.savetxt(outdir + algo_name + '_metrics.csv', np.column_stack((ids, algorithm_metrics)), delimiter=',',
               fmt=['%s', '%f', '%f', '%f', '%f', '%f', '%f'],
               header='img ID, recall, precision, undersegmentation Bergh, undersegmentation NP, compactness, density',
               comments='')

    recall = algorithm_metrics[:, 0]
    precision = algorithm_metrics[:, 1]

    plt.figure(dpi=180)
    plt.plot(np.arange(len(datasets)) + 1, recall, '-o', c='k', label='recall')
    plt.plot(np.arange(len(datasets)) + 1, precision, '-o', c='r', label='precision')
    plt.title(algo_name + ' P/R histogram ' + num_clusters + ' nclusters')
    plt.xlabel(
        'Rmax: %.3f, Rmin: %.3f, Rmean: %.3f, Rmed: %.3f, Rstd: %.3f \n Pmax: %.3f, Pmin: %.3f, Pmean: %.3f, '
        'Pmed: %.3f, Pstd: %.3f ' % (
            recall.max(), recall.min(), recall.mean(), np.median(recall), recall.std(),
            precision.max(), precision.min(), precision.mean(), np.median(precision),
            precision.std()))
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()
    plt.savefig(outdir + algo_name + '_PR_hist_' + num_clusters + '_nclusters.png', bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def prepare_dataset(img_id, image, gabor_features, img_shape):
    ground_truth = np.array(get_segment_from_filename(img_id))
    n_segments = get_num_segments(ground_truth)
    return (img_id, image, gabor_features, ground_truth, n_segments, img_shape), {}


if __name__ == '__main__':
    np.random.seed(0)
    num_cores = -1

    num_imgs = 7

    hdf5_dir = Path('../../data/hdf5_datasets/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_feat = hdf5_dir / 'complete' / 'features'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_feat = hdf5_dir / '7images/' / 'features'
        num_imgs_dir = '7images/'

    print('Reading Berkeley image data set')
    t0 = time.time()
    # Read hdf5 file and extract its information
    images_file = h5py.File(hdf5_indir_im / "Berkeley_images.h5", "r+")
    image_vectors = images_file["images"][:]
    img_shapes = images_file["image_shapes"][:]
    img_ids = images_file["image_ids"][:]

    images = Parallel(n_jobs=num_cores)(
        delayed(np.reshape)(img, (shape[0], shape[1], shape[2])) for img, shape in zip(image_vectors, img_shapes))

    t1 = time.time()
    print('Reading hdf5 image data set time: %.2fs' % (t1 - t0))

    input_files = os.listdir(hdf5_indir_feat)
    for features_input_file in input_files:
        with h5py.File(hdf5_indir_feat / features_input_file, "r+") as features_file:
            print('Reading Berkeley features data set')
            print('File name: ', features_input_file)
            t0 = time.time()
            feature_vectors = features_file["gabor_features"][:]
            feature_shapes = features_file["feature_shapes"][:]

            features = Parallel(n_jobs=num_cores)(
                delayed(np.reshape)(features, (shape[0], shape[1])) for features, shape in
                zip(feature_vectors, feature_shapes))
            t1 = time.time()
            print('Reading hdf5 features data set time: %.2fs' % (t1 - t0))

            iterator = zip(img_ids, images, features, img_shapes)

            datasets = Parallel(n_jobs=num_cores)(
                delayed(prepare_dataset)(im_id, image, feature, shape) for im_id, image, feature, shape in iterator)

            default_base = {'n_clusters': 4, 'n_jobs': -1}

            possible_num_clusters = ['max', 'min', 'mean', 'hmean', 'const']
            algo_name = 'KMeans'

            for num_clusters in possible_num_clusters:
                print('\nComputing %s number of cluster: ' % num_clusters)
                outdir = '../outdir/pixel_level_segmentation/' + num_imgs_dir + algo_name + '/' + features_input_file[
                                                                                                  :-3] + '/' + num_clusters + '_nclusters/'

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                segmentation_metrics = Parallel(n_jobs=num_cores, prefer='processes')(
                    delayed(clustering_segmentation)(i_dataset + 1, dataset, algo_params, num_clusters, algo_name) for
                    i_dataset, (dataset, algo_params) in enumerate(datasets))

                algorithm_metrics = []

                for ii in range(len(datasets)):
                    algorithm_metrics.append((segmentation_metrics[ii]['recall'], segmentation_metrics[ii]['precision'],
                                              segmentation_metrics[ii]['underseg'],
                                              segmentation_metrics[ii]['undersegNP'],
                                              segmentation_metrics[ii]['compactness'],
                                              segmentation_metrics[ii]['density']))

                save_plot_metrics(img_ids, algorithm_metrics, algo_name)