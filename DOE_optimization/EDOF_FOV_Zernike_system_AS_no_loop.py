import tensorflow as tf
import numpy as np
import utilities

"""
In this file, we generate the Flatsocpe PSF based on Angular spectrum method (ASM) with parture and 
Zernike phase modulation.

update (12/13/2019): when calculating the zernike phase, make sure the scope is reusable
update(12/12/2019) : since the Zernike polynomial range from -1 to 1, just confine the coefficients >0
update(11/7/2020): to save memory, only use complex type in the last step
update(11/7/2020): add small windows to avoid alaising issue
update(11/7/2020): non-loop version

Last update: 12/13/2019. YZ
"""

class EDOF_FOV_Zernike_system_AS():
    def __init__(self,
                 wave_length,
                 sensor_distance,
                 sample_distance,
                 num_Zernike_mode,
                 zernike_phase,
                 valid_pixel_num,
                 pad_pixel_num,
                 pixel_size,
                 input_sample_interval,
                 input_sample_depth_num,
                 refractive_idcs, 
                 FOV_test_N, 
                 lateral_shift_interval):
        
        self.wave_length = wave_length
        self.sensor_distance = sensor_distance
        self.sample_distance = sample_distance
        self.num_Zernike_mode = num_Zernike_mode
        self.zernike_phase = zernike_phase
        self.valid_pixel_num = valid_pixel_num # number of valide pixel, without padding
        self.pad_pixel_num = pad_pixel_num # number of padding pixel
        self.pixel_size = pixel_size
        self.input_sample_interval = input_sample_interval
        self.input_sample_depth_num = input_sample_depth_num
        self.refractive_idcs = refractive_idcs
        self.FOV_test_N = FOV_test_N
        self.lateral_shift_interval = lateral_shift_interval

        # calcualte the final simulation field
        self.simu_pixel_num = self.valid_pixel_num + 2 * self.pad_pixel_num
    def _build_phase_mask(self):
        # build the phase profile 
        zernike_coef_ini = np.zeros(self.num_Zernike_mode, dtype=np.float32)  # initialize as zeros
        zernike_coef_tf =  tf.constant(zernike_coef_ini)
        zernike_coef_tf = tf.cast(zernike_coef_tf, tf.float32)
        self.zernike_coef = tf.get_variable(name = "Zernike_coef",
                                        dtype=tf.float32, # reduce the data precision
                                        trainable=True,
                                        initializer=zernike_coef_tf , 
                                        constraint=lambda t: tf.clip_by_value(t, -10, 10)
                                        )

        # generate the phase based on an ideal phase and some zernike mode
        with tf.variable_scope("Forward_model") as scope: # why needs this?
            # lens phase as the base
            x = np.array(list(range(1, self.valid_pixel_num + 1))) - self.valid_pixel_num  / 2
            x = x * self.pixel_size
            Y, X = np.meshgrid(x, x)

            ideal_f = 1 / (1 / self.sample_distance  + 1 / self.sensor_distance) # keep the sign with your matlab version
            phase_modulate = -2 * np.pi / self.wave_length * (np.square(X) + np.square(Y)) / 2/ ideal_f # this initialization generate a spot in the sensor
            # phase_modulate = tf.cast(phase_modulate, tf.float32)


            # a for loop to add zernike mode. avoid the matrix production
            # for i in range(0, self.num_Zernike_mode):
            #     buf = tf.cast(self.zernike_phase[i, :, :], tf.float32)
            #     phase_modulate = phase_modulate + tf.multiply(buf, self.zernike_coef[i])
            phase_modulate = tf.cast(phase_modulate, tf.float32) + \
                                tf.reduce_sum(
                                tf.multiply(
                                    tf.expand_dims(tf.expand_dims(self.zernike_coef, -1), -1), 
                                    tf.cast(self.zernike_phase, tf.float32)) # require zernike phase to be size of (num_Zernike_mode, valid_pixel_num, valid_pixel_num)
                                , 0) 

            # add padding here
            phase_modulate = tf.identity(phase_modulate, "modu_phase") # note this should be a 2D tensor, float32
            paddings = tf.constant([[0, 0], [self.pad_pixel_num, self.pad_pixel_num], [self.pad_pixel_num, self.pad_pixel_num], [0, 0]]) 
            self.phase_modulate = tf.pad(tf.expand_dims(tf.expand_dims(
                                                        phase_modulate,
                                                        0), -1), paddings, "CONSTANT")
            # add aperture mask
            self.aperture_mask = tf.pad(tf.expand_dims(tf.expand_dims(
                                                        np.ones([self.valid_pixel_num, self.valid_pixel_num]),
                                                        0), -1), paddings, "CONSTANT")
            scope.reuse_variables()
            

        # print(self.element.shape.as_list())
    
    def get_psfs(self):
        k = 2 * np.pi / self.wave_length * self.refractive_idcs

        # get the phase mask
        self._build_phase_mask()

        # point source generation, with XYZ shift
        x_axis = np.array(list(range(0, self.simu_pixel_num))) - self.simu_pixel_num / 2
        x_axis = x_axis * self.pixel_size  # field axis
        Y, X = np.meshgrid(x_axis, x_axis)

        # z shift
        assert(np.mod(self.input_sample_depth_num , 2) == 1)
        distances = np.array(list(range(1, self.input_sample_depth_num + 1)))- (self.input_sample_depth_num+1) / 2
        distances = distances * self.input_sample_interval + self.sample_distance
         
        # xy shift
        assert(np.mod(self.FOV_test_N, 2) == 1)
        shift_x =  np.array(list(range(1, self.FOV_test_N  + 1)))- (self.FOV_test_N + 1) / 2
        shift_x = shift_x * self.lateral_shift_interval
        shift_y = shift_x
        # shift_x_ind = np.int(shift_x/ self.pixel_size * self.sensor_distanc / self.sample_distance)
        # shift_y_ind = np.int(shift_y/ self.pixel_size * self.sensor_distanc / self.sample_distance)


        crop_window_size = 200
        input_phase = []
        # print(distances)
        effecive_window_info = []

        # build the test wavefront with XYZ shift
        for shift_x_distance in shift_x: # this loop takes a really long time, since tf is trying to build the graph
            for shift_y_distance in shift_y:
                for distance in distances:
                    #  paraxial spherical source wave
                    curr_phase =  k / 2 / distance * (np.square(X + shift_x_distance) + np.square(Y + shift_y_distance)) # broad cust here
                    # curr_window = np.zeros(self.simu_pixel_num, self.simu_pixel_num)
                    # curr_window[np.int(self.simu_pixel_num / 2) + shift_x_ind - crop_window_size / 2 + 1 : np.int(self.simu_pixel_num / 2) + shift_x_ind + crop_window_size / 2, 
                    #             np.int(self.simu_pixel_num / 2) + shift_y_ind - crop_window_size / 2 + 1 : np.int(self.simu_pixel_num / 2) + shift_y_ind + crop_window_size / 2] = 1
                    shift_x_ind = np.int(shift_x_distance/ self.pixel_size * self.sensor_distance / self.sample_distance)
                    shift_y_ind = np.int(shift_y_distance/ self.pixel_size * self.sensor_distance / self.sample_distance)
                    curr_phase = tf.cast(curr_phase, np.float32)
                    curr_window_info = [shift_x_ind, shift_y_ind]
                    input_phase.append(curr_phase)
                    effecive_window_info.append(curr_window_info)

        input_phase = tf.stack(input_phase, axis=-1)
        input_phase = tf.expand_dims(input_phase, 0) # [1, h, w, channel]
        
        # effecive_window_info = tf.stack(effecive_window_info, axis = -1) # [2, channel]

        # generate multiple PSFs 
        psfs = []
        with tf.variable_scope("Forward_model") as scope: 
            # coded by mask. More efficient way is to directly propagate  the whole batch
            # add phase, use broad cast feature
            all_phase = self.phase_modulate + input_phase # [1, h, w, channel]
            incident_field = utilities.compl_exp_tf(all_phase)

            # add aperture
            self.element = tf.multiply(incident_field, tf.cast(self.aperture_mask, tf.complex64)) # broadcast here
            
            # from mask to sensor, use AS propagation
            sensor_field = utilities.propAS_tf_batch(incident_field,
                                                    self.simu_pixel_num * self.pixel_size, # filed size
                                                    self.simu_pixel_num, # pixel number
                                                    self.wave_length, 
                                                    self.sensor_distance)

            # grab the intensity PSF
            psf_global = tf.square(tf.abs(sensor_field), name='psf_source_3D')

            # crop the PSF based on local positions, save memory
            # psf = tf.multiply(psf_global, tf.cast(curr_window, tf.complex64))

            for source_3D_idx, curr_window_info in enumerate(effecive_window_info): # this for loop takes a lot of times
                psf = psf_global[:, 
                            np.int(self.simu_pixel_num / 2 + curr_window_info[0] - crop_window_size / 2 + 1) : np.int(self.simu_pixel_num / 2 + curr_window_info[0] + crop_window_size / 2+ 1), 
                            np.int(self.simu_pixel_num / 2 + curr_window_info[1] - crop_window_size / 2 + 1 ): np.int(self.simu_pixel_num / 2 + curr_window_info[1] + crop_window_size / 2 + 1), 
                            source_3D_idx]
                psfs.append(psf)  # (Height, width, 1, channels)
            # psf = tf.div(psf, tf.reduce_max(psf, axis=[1, 2], keepdims=True), name='psf_source_3D_idx_%d' % source_3D_idx)

            # record, be aware that tensorboard only accepts 4D tensor
            # utilities.attach_summaries('PSF_source_3D_idx_%d' % source_3D_idx, psf, image=True, log_image=True). Block to save the memory

            # final normalization
            psfs = tf.stack(psfs, axis=-1)
            psfs = tf.div(psfs, tf.reduce_sum(psfs, axis=[1, 2, 3], keepdims=True))

            psfs_shape = psfs.shape.as_list()
            assert(np.size(psfs_shape) == 4)
            scope.reuse_variables()

        return psfs