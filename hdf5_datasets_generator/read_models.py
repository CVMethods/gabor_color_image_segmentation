import os
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed, dump, load
import pdb
from tensorflow.keras.models import Sequential, save_model, load_model


if __name__ == '__main__':
    num_cores = -1

    num_imgs = 7

    source_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
    pixel_slic_model_dir = Path(source_dir + '../../data/models/' + str(num_imgs) + 'images/')
    num_imgs_dir = str(num_imgs) + 'images/'

    input_files = sorted(os.listdir(pixel_slic_model_dir))
    for graph_conf_dir in input_files:
        print('\n##############', graph_conf_dir, '##############\n')
        gabor_conf_dir = sorted(os.listdir(pixel_slic_model_dir / graph_conf_dir))

        for model_file_dir in gabor_conf_dir:
            model_files = sorted(os.listdir(pixel_slic_model_dir / graph_conf_dir / model_file_dir ))

            for mm, model_file_name in enumerate(model_files):
                model_name = model_file_name.split('.')[0]
                ext = model_file_name.split('.')[1]

                print('Loading model: ' + model_name)
                if ext == 'sav':
                    model = load(pixel_slic_model_dir / graph_conf_dir / model_file_dir / model_file_name)
                    if hasattr(model, 'steps'):
                        print('Coefficients:', model[model.steps[0][0]].coef_)
                        # print('Standar Scalar mean:', model[model.steps[0][0]].mean_)
                        # print('Standar Scalar var:', model[model.steps[0][0]].var_)
                        print('\n')
                    elif isinstance(model, np.ndarray):
                        print('Coefficients:', model)
                        # print('Standar Scalar mean:', model[model.steps[0][0]].mean_)
                        # print('Standar Scalar var:', model[model.steps[0][0]].var_)
                        print('\n')
                if ext == 'h5':
                    model = load_model(pixel_slic_model_dir / graph_conf_dir / model_file_dir / model_file_name)
                    print('Coefficients:', model.layers[-1].get_weights())
                    print('\n')


                # outdir = source_dir + '../../outdir/' + \
                #          num_imgs_dir + \
                #          'predicted_gradients/' + \
                #          final_dir + '/' + \
                #          model_name + '/' + \
                #          gradients_input_dir + '/'


