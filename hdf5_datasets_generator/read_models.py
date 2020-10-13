import os
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed, dump, load
import pdb

if __name__ == '__main__':
    num_cores = -1

    num_imgs = 500

    hdf5_dir = Path('../../data/hdf5_datasets/')
    sav_dir = Path('../../data/models/')

    if num_imgs is 500:
        # Path to whole Berkeley image data set
        hdf5_indir_im = hdf5_dir / 'complete' / 'images'
        hdf5_indir_spix = hdf5_dir / 'complete' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / 'complete' / 'gradients'
        sav_indir = sav_dir / 'complete'
        hdf5_outdir = hdf5_dir / 'complete' / 'predicted_gradients'
        num_imgs_dir = 'complete/'

    elif num_imgs is 7:
        # Path to my 7 favourite images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '7images/' / 'images'
        hdf5_indir_spix = hdf5_dir / '7images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '7images' / 'gradients'
        sav_indir = sav_dir / '7images'
        hdf5_outdir = hdf5_dir / '7images' / 'predicted_gradients'
        num_imgs_dir = '7images/'

    elif num_imgs is 25:
        # Path to 25 images from the Berkeley data set
        hdf5_indir_im = hdf5_dir / '25images' / 'images'
        hdf5_indir_spix = hdf5_dir / '25images' / 'superpixels'
        hdf5_indir_grad = hdf5_dir / '25images' / 'gradients'
        sav_indir = sav_dir / '25images'
        hdf5_outdir = hdf5_dir / '25images' / 'predicted_gradients'
        num_imgs_dir = '25images/'

    n_slic_regions = '2000_regions'

    input_files = os.listdir(hdf5_indir_grad)
    for gradients_input_file in input_files:
        input_file_name = gradients_input_file.split('_')
        input_file_name[1] = 'Models'
        input_model_dir = '_'.join(input_file_name)[:-3]
        model_input_files = sorted(os.listdir(sav_indir / input_model_dir / n_slic_regions))
        print('\n##############', '_'.join(input_file_name)[16:-3], '##############\n')
        for mm, model_file_name in enumerate(model_input_files):
                print('Loading model: ' + model_file_name[:-4])
                model = load(sav_indir / input_model_dir / n_slic_regions / model_file_name)
                if hasattr(model[model.steps[1][0]], 'coef_'):
                    print('Coefficients:', model[model.steps[1][0]].coef_)
                    # print('Standar Scalar mean:', model[model.steps[0][0]].mean_)
                    # print('Standar Scalar var:', model[model.steps[0][0]].var_)
                    print('\n')

                elif hasattr(model[model.steps[1][0]], 'coefs_'):
                    # print('Coefficients:', np.mean(model[model.steps[1][0]].coefs_[0], axis=1))
                    print('Coefficients:', model[model.steps[1][0]].coefs_[0][:, -1])
                    print('\n')
