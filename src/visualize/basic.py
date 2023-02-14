from matplotlib import pyplot as plt
from skimage import io
from src.utils import path as path_utils

def plot_count(sub_plot=None):
    t1_count = len(path_utils.T1_ALL_IMAGE_PATH)
    t2_count = len(path_utils.T2_ALL_IMAGE_PATH)

    if(sub_plot):
        sub_plot.bar(['T1', 'T2'], [t1_count, t2_count])
        sub_plot.set_title('Count of T1 and T2')
        sub_plot.set_xlabel('Type')
        sub_plot.set_ylabel('Count')
    else:
        plt.bar(['T1', 'T2'], [t1_count, t2_count])
        plt.title('Count of T1 and T2')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.show()

def plot_dimension(sub_plot=None):
    all_image_width = []
    all_image_height = []
    
    for image_path in path_utils.T1_ALL_IMAGE_PATH:
        width, height = io.imread(image_path).shape
        all_image_width.append(width)
        all_image_height.append(height)
    
    for image_path in path_utils.T2_ALL_IMAGE_PATH:
        width, height = io.imread(image_path).shape
        all_image_width.append(width)
        all_image_height.append(height)
    
    # plot width and height on dot chart
    if(sub_plot):
        sub_plot.scatter(all_image_width, all_image_height)
        sub_plot.set_title('Dimension of T1 and T2')
        sub_plot.set_xlabel('Width')
        sub_plot.set_ylabel('Height')
    else:
        plt.scatter(all_image_width, all_image_height, c='r', marker='.')
        plt.title('Dimension of T1 and T2')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.show()


def plot_image_histogram(sub_plot_t1=None, sub_plot_t2=None):
    # plot image histogram one from T1 and one from T2
    t1_image_path = path_utils.T1_ALL_IMAGE_PATH[0]
    t2_image_path = path_utils.T2_ALL_IMAGE_PATH[0]

    t1_image = io.imread(t1_image_path)
    t2_image = io.imread(t2_image_path)

    if(sub_plot_t1):
        sub_plot_t1.hist(t1_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        sub_plot_t1.set_title('Histogram of T1')
        sub_plot_t1.set_xlabel('Intensity')
        sub_plot_t1.set_ylabel('Count')
    else:
        plt.hist(t1_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        plt.title('Histogram of T1')
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.show()
    
    if(sub_plot_t2):
        sub_plot_t2.hist(t2_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        sub_plot_t2.set_title('Histogram of T2')
        sub_plot_t2.set_xlabel('Intensity')
        sub_plot_t2.set_ylabel('Count')
    else:
        plt.hist(t2_image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
        plt.title('Histogram of T2')
        plt.xlabel('Intensity')
        plt.ylabel('Count')
        plt.show()
    
def plot_batch_image(images):
    # tow row subplot
    fig, axes = plt.subplots(2, len(images) // 2, figsize=(5,5))
    i, j = 0, 0
    for image in images:
        axes[i, j].imshow(image, cmap='gray')
        axes[i, j].set_title('Image {}, {}'.format(i, j))
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        j += 1
        if(j == len(images) // 2):
            i += 1
            j = 0

    # plot title
    fig.suptitle('Batch Image, size - ({}, {}, {})'.format(len(images), images[0].shape[0], images[0].shape[1]))
    
    plt.show()

def plot_from_paths(image_paths):
    # plot batch image
    images = []
    for image_path in image_paths:
        images.append(io.imread(image_path))
    
    plot_batch_image(images)

def plot_t1_image(batch_size=6):
    # plot image from T1 and T2
    t1_image_paths = path_utils.T1_ALL_IMAGE_PATH[:batch_size]

    plot_from_paths(t1_image_paths)

def plot_t2_image(batch_size=6):
    # plot image from T1 and T2
    t2_image_paths = path_utils.T2_ALL_IMAGE_PATH[:batch_size]

    plot_from_paths(t2_image_paths)
