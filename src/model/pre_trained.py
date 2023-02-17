from pathlib import Path
import zipfile
import os
import tensorflow as tf
from urllib.request import urlretrieve


from src.utils.path import MODEL_URL

class DLProgbar:
    def __init__(self):
        pass

    def __call__(self, block_num, block_size, total_size):
        current = (block_num * block_size) / total_size * 100
        current = int(min(round(current, 2), 100))

        total_dots = 30
        total_cross = int(current / 100 * total_dots)
        total_dots = total_dots - total_cross

        print(f'\rDownloading model |{"*"*int(total_cross)}{"."*total_dots}| [{str(current).zfill(3)}% / 100.00%]', end='')

        if current == 100:
            print(' - Done')


def download_model(retry = 0):
    """
    Download model from google drive
    """
    
    home = Path.home()

    path_to_model = home / '.mrigan'

    if not path_to_model.exists():
        path_to_model.mkdir()
    
    model_location = path_to_model / 'models.zip'
    model_dir = path_to_model / 'models'

    if not model_dir.exists():
        model_dir.mkdir()

    # download model
    if not model_location.exists():
        try:
            urlretrieve(MODEL_URL, model_location, DLProgbar())
        except Exception as e:
            print('Error', e)
            os.remove(model_location)

    if len(os.listdir(model_dir)) < 2:
        try:
            # unzip model
            print('Unzipping model...', end=' ')
            with zipfile.ZipFile(model_location, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            print('Done')
        except zipfile.BadZipFile as e:
            print('Error', e)
            os.remove(model_location)
            if retry < 3:
                download_model(retry + 1)

    return model_dir

def load_model(model_type='T1'):
    """
    Load model
    """

    path_to_model = download_model()

    print('Loading model...', end=' ')

    if model_type not in ['T1', 'T2']:
        raise ValueError('model_type must be T1 or T2')

    if model_type == 'T1':
        path_to_model = path_to_model / 'mri_gan_t1_to_t2'
    else:
        path_to_model = path_to_model / 'mri_gan_t2_to_t1'
    
    model = tf.keras.models.load_model(str(path_to_model))

    print('Done')

    return model