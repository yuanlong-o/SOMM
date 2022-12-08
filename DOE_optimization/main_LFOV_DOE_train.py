import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # only in cpu

import model
import utilities
import L2_deconv
import EDOF_FOV_Zernike_system_AS_no_loop
import data_reader

import numpy as np
import tensorflow as tf

from glob import glob
import h5py

"""
In this file, we run single mask flatscope with aperture consideration and Zernike polynomial incident. 
Different from reconstruction task, we seek extended depth of field along with large FOV in this file.
We here use AS propagation instead of Fresnel propagation to model the FOV issue

note in this version the FOV test can not be 
update: 



Last update: 11/07/2020. YZ
"""

class EDOF_FOV_Flatscope_model(model.Model):
    def __init__(self,
                wave_length,
                sensor_distance,
                sample_distance,
                num_Zernike_mode,
                zernike_phase,
                valid_pixel_num,
                pad_pixel_num,
                input_sample_pixel,
                pixel_size,
                input_sample_interval,
                input_sample_depth_num,
                refractive_idcs,
                FOV_test_N, 
                lateral_shift_interval,
                batch_size,
                ckpt_path,
                target_dir):

        self.wave_length = wave_length
        self.sensor_distance = sensor_distance
        self.sample_distance = sample_distance
        self.num_Zernike_mode = num_Zernike_mode
        self.zernike_phase = zernike_phase
        self.valid_pixel_num = valid_pixel_num # number of valide pixel, without padding
        self.pad_pixel_num = pad_pixel_num # number of padding pixel

        self.input_sample_pixel = input_sample_pixel
        self.pixel_size = pixel_size
        self.input_sample_interval = input_sample_interval
        self.input_sample_depth_num = input_sample_depth_num
        self.refractive_idcs = refractive_idcs
        self.FOV_test_N = FOV_test_N
        self.lateral_shift_interval = lateral_shift_interval


        self.batch_size=batch_size
        self.target_dir = target_dir
        # run the heritance initialization
        super(EDOF_FOV_Flatscope_model, self).__init__(name='EDOF_FOV_Flatscope_model', ckpt_path=ckpt_path)
    
    # _build_graph method, get the output
    def _build_graph(self, x_train, global_step, init_gamma):
        # x_train: [batch_size, height, width]

        input_img = x_train
        # for razer blade, use GPU 1
        # for GT75, use GPU 0
        with tf.device('/device:GPU:1'):
            # get PSF
            Flatscop = EDOF_FOV_Zernike_system_AS_no_loop.EDOF_FOV_Zernike_system_AS(
                            wave_length = self.wave_length,
                            sensor_distance = self.sensor_distance,
                            sample_distance = self.sample_distance,
                            num_Zernike_mode = self.num_Zernike_mode,
                            zernike_phase = self.zernike_phase,
                            valid_pixel_num = self.valid_pixel_num,
                            pad_pixel_num = self.pad_pixel_num,
                            pixel_size = self.pixel_size,
                            input_sample_interval = self.input_sample_interval,
                            input_sample_depth_num = self.input_sample_depth_num,
                            refractive_idcs = self.refractive_idcs,                 
                            FOV_test_N = self.FOV_test_N, 
                            lateral_shift_interval = self.lateral_shift_interval)

        # get PSF based on system
        psfs = Flatscop.get_psfs() # [1, Height, width, channels]
        gamma_ini = init_gamma * np.ones([1, input_sample_depth_num * FOV_test_N * FOV_test_N, 1, 1], dtype=np.float32)  # initialize as zeros
        gamma_tf =  tf.constant(gamma_ini)
        gamma_tf = tf.cast(gamma_tf, tf.float32)
        gamma = tf.get_variable(name = "gamma",
                                        dtype=tf.float32, # reduce the data precision
                                        trainable=True,
                                        initializer=gamma_tf, 
                                        constraint=lambda t: tf.clip_by_value(t, 0, 1e4)
                                        )
        # do PSF-wared observation        
        cap_image = L2_deconv.gen_observ(input_img, psfs) # beaware here input imag shall be a 2D image, output is [batch_size, num_psfs, height, width]

        # do deconvolution, use central PSF
        psf_center = tf.squeeze(tf.slice(psfs, [0, 0, 0, int(self.input_sample_depth_num * self.FOV_test_N * self.FOV_test_N /2)], [-1, -1, -1, 1])) # [height, width]
        psf_center = tf.expand_dims(psf_center, 0) # [1, height, width]

        output_image = L2_deconv.deconv_L2_per_depth_layer(cap_image, psf_center, gamma) # last for gamma, output size [batch_size, num_psfs, height, width]
            
        return psfs, output_image, gamma

    # _get_data_loss method, calculate the data loss
    def _get_data_loss(self, model_output, ground_truth, psfs, margin = 10):
        # model_output: [batch_size, num_psfs, height, width]
        # ground_truth: [batch_size, height, width]
        # psfs: [1, height, width, num_psfs]

        # be aware here ground_truth is a 2d image instead of a 3D image
        model_output = tf.cast(model_output, tf.float32)
        ground_truth = tf.cast(ground_truth, tf.float32)
        ground_truth = tf.expand_dims(ground_truth, 1) #  [batch_size, 1, height, width]
        # curr_model_shape = model_output.shape.as_list() 
        # tile
        # ground_truth = tf.tile(ground_truth, tf.constant([1, curr_model_shape[1], 1, 1])) # use tile to copy the same 2d samples

        # scale to sum of all pixels in lateral palne.
        model_output = tf.div(model_output, tf.reduce_sum(model_output, axis=[2, 3], keepdims=True))
        ground_truth = tf.div(ground_truth, tf.reduce_sum(ground_truth, axis=[2, 3], keepdims=True))
        
        # MSE loss, sensitive to shift
        # loss = tf.reduce_mean(tf.square(model_output - ground_truth)[:, :, margin:-margin,margin:-margin]) * 1e5 # directly broadcast

        # do convolution, then calculate the maximum of convolution results (based on its principle), then average different positions
        ground_truth = tf.transpose(ground_truth, [0, 2, 3, 1])  # [batch_size,height, width, 1]
        ground_truth = tf.expand_dims(ground_truth, 0) # add dummy dimension, [1, batch_size,height, width, 1], now [1, batch_size, height,width] as spatial dimension
        model_output  = tf.transpose(model_output, [0, 2, 3, 1]) # [batch_size,height, width, num_psfs]
        # model_output = tf.reverse(model_output, [1, 2]) # no need for reverse
        model_output = tf.expand_dims(model_output, 3) # add dummy dimension for input filter size, now it is  [batch_size, height, width, 1, num_psfs], N+2
        conv_out = tf.nn.convolution(ground_truth[:, :, margin:-margin,margin:-margin, :], model_output[:, margin:-margin,margin:-margin, :, :], 
                                    strides=None, padding='SAME') # [1, batch_size, height, width, num_psfs]
        conv_out = tf.squeeze(conv_out, axis=0) # remove dummy batch size, now is [batch_size, height, width, num_psfs] 
        loss1 = -tf.reduce_mean(tf.reduce_max(conv_out, axis=[1, 2], keepdims=False)) # as large as possible

        # define second loss for uniformity of energy in different depth
        loss2 = tf.squeeze(tf.reduce_max(psfs, axis=[1, 2]))
        loss2 = tf.reduce_mean(tf.square(loss2 - tf.reduce_mean(loss2)))
        return loss1, loss2 # force uniform

    def _get_training_queue(self, batch_size):
        # read image
        image_batch = data_reader.get_training_data(self.target_dir,
                                                    self.input_sample_pixel,
                                                    self.batch_size,
                                                    self.input_sample_depth_num,
                                                    True)
        image_batch_MIP = tf.reduce_sum(image_batch, axis=3)

        # # single image mode
        # image_batch = data_reader.get_test_data(self.target_dir,
        #                                         self.pixel_resolution,
        #                                         1,
        #                                         self.input_sample_depth_num,
        #                                         True)
        return image_batch_MIP, image_batch_MIP # here we just need 2d images, size [batch_size, height, width]

    # test part
    def _get_test_queue(self):
        # read image
        image_batch = data_reader.get_test_data(self.target_dir,
                                                self.input_sample_pixel,
                                                1,
                                                self.input_sample_depth_num,
                                                True)
        image_batch_MIP = tf.reduce_sum(image_batch, axis=3)
        return image_batch_MIP


    def test(self, psfs, x_test, gamma):
        # psfs: [Height, width, 1, channels]
        # x_test: [batch_size, height, width]
        cap_image = L2_deconv.gen_observ(x_test, psfs)

        # slice 
        psf_center = tf.squeeze(tf.slice(psfs, [0, 0, 0, int(self.input_sample_depth_num * self.FOV_test_N * self.FOV_test_N /2)], [-1, -1, -1, 1]))
        psf_center = tf.expand_dims(psf_center, 0)

        # per-dpeth deconvolution
        return L2_deconv.deconv_L2_per_depth_layer(cap_image, psf_center, gamma) # output will be [batch_size, num_psfs, height, width]

if __name__=='__main__':
    tf.reset_default_graph()
    # set threads number
    tf.config.threading.set_inter_op_parallelism_threads(
        num_threads=36
    )

    # Note the simulation parameter should be carefully chosen considering sampling. Check the .m file for guide
    wave_length = 532e-9
    refractive_idcs = 1 # refractive index in the propagation medium

    ## distance and size parameters
    sensor_distance = 8e-3
    sample_distance = 8e-3

    valid_pixel_num = 750
    pad_pixel_num = 500 # note this is the pad pixel in one side
    pixel_size = 2e-6

    # information about the test sample, in meters
    input_sample_pixel = 128
    input_sample_interval = 150e-6
    input_sample_depth_num = 3
    input_sample_lateral_size = 9 # neuron later size

    # FOV testing
    max_FOV_shift = 1.5e-3 # half size of target FOV
    FOV_test_N = 3 # number of FOV that is used for test
    lateral_shift_interval = max_FOV_shift

    num_Zernike_mode = 100

    
    batch_size = 1
    ckpt_path = None
    # sample path
    N = 10
    target_dir = 'PATH_TO_SAMPLE/size_{0:.0f}_{1:.0f}_{2:.0f}_N_{3:.0f}_lateral_{4:.0f}'.format(input_sample_pixel, 
                                    input_sample_pixel, input_sample_depth_num, N, input_sample_lateral_size)

    # Zernike wavefront
    f = h5py.File('PATH_TO_ZERNIKE/Zernike_max_{0:.0f}_eff_size_{1:.0f}.mat'.format(100, valid_pixel_num),'r')
    zernike_phase = f.get('z_array')
    zernike_phase = zernike_phase[:num_Zernike_mode, :, :]
    # zernike_phase = np.float32(zernike_phase)

    num_steps = 250001
    # logdir
    log_dir = os.path.join('./log', 'lens_init_RTX3090', 'mode_{0}_valid_{1}_step{2}_dx_{3:.2f}_z1_{4:.1f}_z2_{5:.1f}_psf_loss'.format(num_Zernike_mode, # dont make the name too long!
                            valid_pixel_num, num_steps, pixel_size * 1e6, sample_distance * 1e3,  sensor_distance * 1e3))
    # log_dir = os.path.join('.\log', 'single_distribution')
    if not os.path.exists(log_dir): 
        os.makedirs(log_dir, exist_ok=True)

    ## write parameters
    File_object = open(r'{0}/options'.format(log_dir), "a")
    File_object.write("wave_length" + " " + str(wave_length) + "\n")
    File_object.write("refractive_idcs" + " " + str(refractive_idcs) + "\n")
    File_object.write("sensor_distance" + " " + str(sensor_distance) + "\n")
    File_object.write("pad_pixel_num" + " " + str(pad_pixel_num) + "\n")
    File_object.write("valid_pixel_num" + " " + str(valid_pixel_num) + "\n")
    File_object.write("wave_length" + " " + str(wave_length) + "\n")
    File_object.write("pixel_size" + " " + str(pixel_size) + "\n")
    File_object.write("input_sample_pixel" + " " + str(input_sample_pixel) + "\n")
    File_object.write("input_sample_interval" + " " + str(input_sample_interval) + "\n")
    File_object.write("input_sample_depth_num" + " " + str(input_sample_depth_num) + "\n")
    File_object.write("input_sample_lateral_size" + " " + str(input_sample_lateral_size) + "\n")
    File_object.write("max_FOV_shift" + " " + str(max_FOV_shift) + "\n")
    File_object.write("FOV_test_N" + " " + str(FOV_test_N) + "\n")
    File_object.write("num_Zernike_mode" + " " + str(num_Zernike_mode) + "\n")
    File_object.write("batch_size " + " " + str(batch_size ) + "\n")
    File_object.close()

    edof_fov_flatscope_model = EDOF_FOV_Flatscope_model(
                 wave_length = wave_length,
                 sensor_distance = sensor_distance,
                 sample_distance = sample_distance,
                 num_Zernike_mode = num_Zernike_mode,
                 zernike_phase = zernike_phase,
                 valid_pixel_num = valid_pixel_num,
                 pad_pixel_num = pad_pixel_num,
                 pixel_size = pixel_size,
                 input_sample_pixel = input_sample_pixel,
                 input_sample_interval = input_sample_interval,
                 input_sample_depth_num = input_sample_depth_num,
                 refractive_idcs = refractive_idcs,
                 FOV_test_N = FOV_test_N, 
                 lateral_shift_interval = lateral_shift_interval,
                 batch_size = batch_size,
                 ckpt_path = ckpt_path,
                 target_dir = target_dir)

    # optimize parameter
    # opt_params_for_ADAM ={'epsilon' : 1e-5}
    # learning rate for 1um pixel size: 2e-3 is OK

    edof_fov_flatscope_model.fit(model_params={'init_gamma' : 1e1},
                  opt_type = 'ADAM',
                  opt_params = {},
                  batch_size=batch_size,
                  starter_learning_rate=2e-3,
                  num_steps_until_save=5000,
                  num_steps_until_summary=5000,
                  logdir = log_dir,
                  num_steps = num_steps)
