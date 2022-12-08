import tensorflow as tf
import numpy as np
import utilities as utilities

def deconv_L2_per_depth_layer(observ, psf_center, gamma):
    """Implements L2 deconvolution for single planes. Use FFT based algorithm to boost the performance
        we abandon the requirement that the image and the PSF should have the same size.
        last update: 11/07/2020. YZ

    Args:
         observ: generated observation [batch, num_psfs, height_after_padding, width_after_padding]. Note the observation is alredy padded
         psf_center: filters with shape [1, height, width]
         gamma: L2 regularizor

    Return:
        result: deconvoled stack with shape [batch_size, num_psfs, height, width]
    Update:
    align the sum of PSF with sample
    reuse the variable to save the GPU memory
    """
    # same data type
    observ = tf.cast(observ, tf.float32)
    psf_center = tf.cast(psf_center, tf.float32)
    psf_center = tf.expand_dims(psf_center, 1) #[1, 1, h, w]
    observ_shape = observ.shape.as_list() # with shape of (batch_size, height, width, num_img_depths), this is not a tensor
    psf_shape = psf_center.shape.as_list()

    # assert height, width, channel size are the same
    # assert (len(observ_shape) == len(psf_shape))

    # paddings
    height_psf_pad = int(observ_shape[2] / 2 - psf_shape[2]/2)
    width_psf_pad = int(observ_shape[3] / 2- psf_shape[3]/2)

    height_sample_pad = int(psf_shape[2]/2)
    width_sample_pad = int(psf_shape[3]/2)

    paddings_psf = tf.constant([[0, 0], [0, 0], [height_psf_pad, height_psf_pad], [width_psf_pad, width_psf_pad]]) # note here paddings_psf is 4D [1, 1, pad_h, pad_w]

    # initialize gamma, in position-wise manner
    # gamma = tf.ones([1, len(observ_shape), 1, 1])
    gamma = tf.cast(gamma, tf.complex64)

    # align the total summation of PSF and sample
    psf_center = tf.div(psf_center, tf.reduce_sum(psf_center, axis=[2, 3], keepdims=True)) * observ_shape[2] * observ_shape[3] # broadcast here
    observ = tf.div(observ, tf.reduce_sum(observ, axis=[2, 3], keepdims=True)) * observ_shape[2] * observ_shape[3]

    # add a dummy depth channel to fit the FFT function
    # observ = tf.expand_dims(observ, -1)

    # in frequency domain
    psf_center =  tf.fft2d(  # note here we reuse the name
                tf.cast(
                    utilities.ifftshift2d_tf_most_inner(tf.pad(psf_center, paddings_psf, "CONSTANT")),
                    tf.complex64)
                ) #  now the dimension should be (1, 1, pad_h, pad_w)

    HtH = tf.multiply(psf_center, tf.math.conj(psf_center))

    denominator = HtH + gamma # (1,  num_channel, pad_h, pad_w)
    
    # generate observation in frequency
    observ_s = tf.fft2d( # why it becomes a special variable
                tf.cast(
                observ, tf.complex64)
                ), # (batch_size, num_channels, height, width)

    # observ = tf.math.real(tf.ifft2d(observ_s))
    # observ = observ[:, height_pad : -height_pad, width_pad : -width_pad]

    # do L2 deconvolution

    # print(psf_transp_f_conj.shape.as_list())
   
    numerator = tf.multiply(observ_s, tf.math.conj(psf_center))
    if (len(numerator.shape.as_list())==int(5)): # some times it add one dummy dimension at the first place
        numerator = tf.squeeze(numerator, 0)

    result = tf.math.real(tf.ifft2d(
            tf.div(numerator, denominator)
            ))
    # result = tf.squeeze(result)

    # remove the pad
    result = result[:, :, height_sample_pad : -height_sample_pad, width_sample_pad : -width_sample_pad] 
    # result [result  < 0] = 0
    result = tf.nn.relu(result)
    return result # return result would be (batch_size, num_channel, sample height, sample width)

# 
def gen_observ(sample, psf):
    """Implements perdepth convolution and summation. Use FFT based algorithm to boost the performance
         we abandon the requirement that the image and the PSF should have the same size.

        last update: 11/7/2020. YZ

    Args:
         sample: generated sample [batch_size, height, width], only a 2D sample is needed!
         psf: filters with shape [1, height, width, num_test_positions]

    Return:
        observ: deconvoled stack with shape (batch_size, height, width, depth)
    Update:
    """
    # same data type
    sample = tf.cast(sample, tf.float32)
    psf = tf.cast(psf, tf.float32)
    psf_shape = psf.shape.as_list()

    # expand the sample
    sample = tf.expand_dims(sample, -1)
    sample = tf.tile(sample, tf.constant([1, 1, 1, psf_shape[-1]])) # use tile to copy the same 2d samples

    sample_shape = sample.shape.as_list() # with shape of (batch_size, height, width, num_img_depths), this is not a tensor
    

    # assert height, width, channel size are the same
    assert (len(sample_shape) ==len(psf_shape))

    # paddings
    height_sample_pad = int(psf_shape[1] / 2)
    width_sample_pad = int(psf_shape[2] / 2)
    height_psf_pad = int(sample_shape[1] / 2)
    width_psf_pad = int(sample_shape[2] / 2)

    paddings_sample = tf.constant([[0, 0], [0, 0], [height_sample_pad, height_sample_pad], [width_sample_pad, width_sample_pad]])
    paddings_psf = tf.constant([[0, 0], [0, 0], [height_psf_pad, height_psf_pad], [width_psf_pad, width_psf_pad]])

    # align the total summation of PSF and sample
    psf = tf.div(psf, tf.reduce_sum(psf, axis=[1, 2], keepdims=True)) * psf_shape [1] * psf_shape [2] # broadcast 
    sample = tf.div(sample, tf.reduce_sum(sample, axis=[1, 2], keepdims=True)) * psf_shape [1] * psf_shape [2] 

    # permute the tensor, everything has shape (batch_size, num_channels, height, width)
    sample_transp = tf.transpose(sample, [0, 3, 1, 2])
    psf_transp = tf.transpose(psf, [0, 3, 1, 2])

    # in frequency domain
    psf_transp_f =  tf.fft2d(
                tf.cast(
                    utilities.ifftshift2d_tf_most_inner(tf.pad(psf_transp, paddings_psf, "CONSTANT"))
                 , tf.complex64)
                )

    # generate observation
    observ_s = tf.multiply(psf_transp_f, # Note this multiply is broadcast, since
                tf.fft2d(
                tf.cast(
                tf.pad(sample_transp, paddings_sample, "CONSTANT")
                , tf.complex64)
                )
                )

    observ = tf.math.real(tf.ifft2d(observ_s)) # note fft and ifft will always work in the most inner two dimensions
    # observ[observ < 0] = 0
    observ = tf.nn.relu(observ)
    return observ # (batch_size, num_channels, height_after_padding, width_after_padding), note the size is still with padding

