from pathlib import Path
from glob import glob
import random

# get root path of project
ROOT_PATH = Path(__file__).parent.parent.parent

# get path of data
DATA_PATH = ROOT_PATH / 'data'

# get path of T1
T1_PATH = DATA_PATH / 'T1'

# get path of T2
T2_PATH = DATA_PATH / 'T2'

# get all image path of T1
T1_ALL_IMAGE_PATH = glob(str(T1_PATH / '*.png'))

# get all image path of T2
T2_ALL_IMAGE_PATH = glob(str(T2_PATH / '*.png'))

# if number of T1 and T2 is not equal, then choose random image path of T1 or T2 and append to list
if len(T1_ALL_IMAGE_PATH) != len(T2_ALL_IMAGE_PATH):
    if len(T1_ALL_IMAGE_PATH) > len(T2_ALL_IMAGE_PATH):
        for _ in range(len(T1_ALL_IMAGE_PATH) - len(T2_ALL_IMAGE_PATH)):
            T2_ALL_IMAGE_PATH.append(random.choice(T2_ALL_IMAGE_PATH))
    else:
        for _ in range(len(T2_ALL_IMAGE_PATH) - len(T1_ALL_IMAGE_PATH)):
            T1_ALL_IMAGE_PATH.append(random.choice(T1_ALL_IMAGE_PATH))