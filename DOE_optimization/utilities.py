"""
this file record the utilitied functions for whole optimziation. Including:
fftshift, ifftshift
Fresnel propagation



last update: 12/2/2019. YZ
"""

import tensorflow as tf
import numpy as np

## define transpose FFT2D and IFFT2D
def transp_fft2d_tf(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d


def transp_ifft2d_tf(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

## define Fresnel propagation, 4D
def propTF_tf_batch(U1, L, M, wavelength, z, dtype=tf.complex64, name='propTF'):
    # U1: input light field, 4D
    # L: size of the side
    # wavelength
    # z: propagation distance
    U1 = tf.cast(U1, tf.complex64)
    fx = (np.array(list(range(1, M + 1))) - M /2) / L
    FY, FX = np.meshgrid(fx, fx)

    # permute

    H = np.exp(-1j * np.pi * wavelength * z * (np.square(FX) + np.square(FY)))
    H = tf.cast(H, tf.complex64)
    # expand dimension
    H = tf.expand_dims(tf.expand_dims(H, 0), -1)
    
    H_1 = fftshift2d_tf(H)
    U1_2 = transp_fft2d_tf(fftshift2d_tf(U1))
    U1_3 = tf.multiply(U1_2, H_1)
    U1_4 = ifftshift2d_tf(transp_ifft2d_tf(U1_3))
    return U1_4

## define Fresnel propagation, 4D
def propAS_tf_batch(U1, L, M, wavelength, z, dtype=tf.complex64, name='propTF'):
    # U1: input light field, 4D, size [batch, height, width, channel]
    # L: size of the side
    # wavelength
    # z: propagation distance
    U1 = tf.cast(U1, tf.complex64)
    fx = (np.array(list(range(1, M + 1))) - M /2) / L
    FY, FX = np.meshgrid(fx, fx)

    w1 = np.square(1/wavelength) - np.square(FX) - np.square(FY)
    w1[w1 < 0] = 0
    w1 = np.sqrt(w1)
    # permute

    H = np.exp(1j * 2 *  np.pi * w1 * z )
    H = tf.cast(H, tf.complex64)
    # expand dimension
    H = tf.expand_dims(tf.expand_dims(H, 0), -1)
    
    H_1 = fftshift2d_tf(H)
    U1_2 = transp_fft2d_tf(fftshift2d_tf(U1))
    U1_3 = tf.multiply(U1_2, H_1)
    U1_4 = ifftshift2d_tf(transp_ifft2d_tf(U1_3))
    return U1_4

## from phase to complex amplitude
def compl_exp_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)

## fftshift2d, dimension in the center
def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3): 
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

## ifftshift2d, dimension in the center
def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

## fftshift2d, most inner 2 dimension
def fftshift2d_tf_most_inner(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(2, 4): 
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

## ifftshift2d, most inner 2 dimension
def ifftshift2d_tf_most_inner(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(2, 4):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

## attach_summarizes
def attach_summaries(name, var, image=False, log_image=False):
    '''Attaches summaries to var.

    :param name: Name of the variable in tensorboard.
    :param var: Variable to attach summary to.
    :param image: Whether an image summary should be created for var.
    :param log_image: Whether a log image summary should be created for var.
    :return:
    '''

    tf.summary.scalar(name + '_mean', tf.reduce_mean(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))
    tf.summary.histogram(name + '_max', tf.reduce_max(var))

    if image:
        tf.summary.image(name, var, max_outputs=3)
    if log_image and image:
        tf.summary.image(name + '_log', tf.log(var + 1e-12), max_outputs=3)

        var_min = tf.reduce_min(var, axis=(1, 2), keepdims=True)
        var_max = tf.reduce_max(var, axis=(1, 2), keepdims=True)
        var_norm = (var - var_min) / (var_max - var_min)
        tf.summary.image(name + '_norm', var_norm, max_outputs=3)