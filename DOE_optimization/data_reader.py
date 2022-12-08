import numpy as np
import tensorflow as tf
import scipy.io
import h5py
import tensorflow as tf
import os

from glob import glob

#  read the image data from mat file
#  last update: 12/10/2019. YZ
#  update: use reference for larger date input

# simple version, directly take the data into a tensor
def get_training_data(target_dir,
                           patch_size,
                           batch_size,
                           num_depths=9,
                           repeat=True):
    '''Data reader that ingests images and cuts out patches with patch_size.

    :param target_dir (str): Directory containing images.
    :param patch_size (str): Size of the patches to be extracted from images.
    :param batch_size: Batch size.
    :param num_depths: Number of depth planes that a single image is placed at.
    :param repeat: Whether to loop the dataset.
    :return:
    Dataset type iterator
    dimension out (batch, patch_size, patch_size, num_depth )
    '''

    # this is a naive implementation
    # patch_size =128
    # batch_size =128
    # num_depths = 9
    # repeat = True
    # N = 10
    # target_dir = '.\data\size_{0:.0f}_{1:.0f}_{2:.0f}_N_{3:.0f}'.format(patch_size, patch_size, num_depths, N)

    mat_file_path = '{}\\train_sample_array.mat'.format(target_dir)
    f = h5py.File(mat_file_path,'r')
    sample_data = f.get('train_sample_array')
    # revser because of mat file
    sample_data = np.transpose(sample_data, [3, 2, 1 ,0])
    sample_data = sample_data[:100, :, :, :] # reduce memory cost

    # operation
    img_dataset = tf.data.Dataset.from_tensor_slices(sample_data)
    img_dataset = img_dataset.shuffle(1000)
    if repeat:
        img_dataset = img_dataset.repeat()
    img_dataset = img_dataset.batch(batch_size)
    img_dataset = img_dataset.prefetch(4*batch_size)

    image_batch = img_dataset.make_one_shot_iterator().get_next()
    image_batch.set_shape((batch_size, patch_size, patch_size, num_depths))

    return image_batch

# simple version, directly take the data into a tensor
def get_test_data(target_dir,
                           patch_size,
                           batch_size,
                           num_depths=9,
                           repeat=True):
    '''Data reader for single stack.

    Dataset type iterator
    Dataset type iterator
    dimension out (batch, patch_size, patch_size, num_depth )
    '''

    # this is a naive implementation
    # patch_size =128
    # batch_size =128
    # num_depths = 9
    # repeat = True
    # N = 10
    # target_dir = '.\data\size_{0:.0f}_{1:.0f}_{2:.0f}_N_{3:.0f}'.format(patch_size, patch_size, num_depths, N)

    mat_file_path = '{}\\test_sample_array.mat'.format(target_dir)
    f = h5py.File(mat_file_path,'r')
    sample_data = f.get('test_sample_array')
    
    sample_data = np.transpose(sample_data, [3, 2, 1, 0])
    # operation
    img_dataset = tf.cast(sample_data, tf.float32)

    return img_dataset