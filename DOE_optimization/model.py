import os
from pprint import pprint

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import utilities
import numpy as np
import abc
import scipy.io
import tifffile


"""
Class define file. We define the a Model class with fitting and storage option.
data grab, loss define needs to be further defined.
Code is taken from VINCENT SITZMANN and modified by YZ.

update: add gpu option, to avoid the thread waiting for empty buffer
update: add tfdebug
update: add thread coordinate system.
update: use customized ConfigProto
update: freeze the graph before the training.
udpate: disable the summary writer. It would break the program I guess?
update: add addtional output for _build_graph, for test
update: add test in summary for tensorboard, and also save it as a mat file.
update: add tiff file writer



last update: 12/13/2019. YZ
"""

slim = tf.contrib.slim
class Model(abc.ABC):
    """Generic tensorflow model class.
    """
    def __init__(self, name, ckpt_path=None, tfdebug=False):
        gpu_options = tf.GPUOptions(polling_inactive_delay_msecs = 10)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.tfdebug = tfdebug
        self.name = name
        self.ckpt_path = ckpt_path

    @abc.abstractmethod
    def _build_graph(self, x_train, global_step, **kwargs): 
        """Builds the model, given x_train as input.

        Args:
            x_train: The dequeued training example
            **kwargs: Model parameters that can later be passed to the "fit" function, cool!

        Returns:
            model_output: The output of the model
        """

    @abc.abstractmethod
    def _get_data_loss(self,
                      model_output,
                      ground_truth):
        """Computes the data loss (not regularization loss) of the model.

        For consistency of weighing of regularization loss vs. data loss,
        normalize loss by batch size.

        Args:
            model_output: Output of self._build_graph
            ground_truth: respective ground truth

        Returns:
            data_loss: Scalar data loss of the model.         """


    def _get_reg_loss(self):
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return reg_loss


    @abc.abstractmethod
    def _get_training_queue(self, batch_size):
        """Builds the queues for training data.

        Use tensorflow's readers, decoders and tf.train.batch to build the dataset.

        Args:
            batch_size:

        Returns:
            x_train: the dequeued model input
            y_train: the dequeued ground truth
        """
    def _set_up_optimizer(self,
                          starter_learning_rate,
                          decay_type,
                          decay_params,
                          opt_type,
                          opt_params,
                          global_step):
        if decay_type is not None:
            if decay_type == 'exponential':
                learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
            elif decay_type == 'polynomial':
                learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                           global_step,
                                                           **decay_params)
        else:
            learning_rate = starter_learning_rate

        if opt_type == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               **opt_params)
        elif opt_type == 'sgd_with_momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'Adadelta' or opt_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  **opt_params)
        else:
            raise Exception('Unknown opt type')

        # tf.summary.scalar('learning_rate', learning_rate)
        return optimizer

    ## test part
    @abc.abstractmethod
    def _get_test_queue(self):
        """Builds the queues for test data.
        """

    @abc.abstractmethod
    def test(self, psfs, x_test, **kwargs):
        """
        Show the deconvolution performances in one test data
        """

    # training operation
    def fit(self,
            model_params, # Dictionary of model parameters
            opt_type, # Type of optimization algorithm
            opt_params, # Parameters of optimization algorithm
            batch_size,
            starter_learning_rate,
            logdir,
            num_steps,
            num_steps_until_save,
            num_steps_until_summary,
            num_steps_until_val=None,
            x_val_list=None,
            decay_type=None, # Type of decay
            decay_params=None, # Decay parameters
            feed_dict=None,
            ):
        """Trains the model.
        """
        x_train, y_train = self._get_training_queue(batch_size)
        x_test = self._get_test_queue()

        print("\n\n")
        print(40*"*")
        print("Saving model and summaries to %s"%logdir)
        print("Optimization parameters:")
        print(opt_type)
        print(opt_params)
        print("Starter learning rate/s is/are ",starter_learning_rate)
        print("Model parameters:")
        print(model_params)
        print(40*"*")
        print("\n\n")

        global_step = tf.Variable(0, trainable=False) # why it need a global step?

        # grab psfs
        psfs, model_output_train, gamma = self._build_graph(x_train, global_step, **model_params) # note the usage of model_params here

        # grab data loss
        data_loss_graph1, data_loss_graph2 = self._get_data_loss(model_output_train, y_train, psfs)
        data_loss_graph = data_loss_graph1 + data_loss_graph2 * 1000
        # regularization loss
        reg_loss_graph = self._get_reg_loss()
        total_loss_graph = tf.add(reg_loss_graph,
                                  data_loss_graph)

        # output test 
        test_result = self.test(psfs, x_test, gamma) # note the dimension shoule be (1, height, width)

        # grab the phase
        var_phase = tf.get_default_graph().get_tensor_by_name("Forward_model/modu_phase:0")
        var_coef = [v for v in tf.global_variables() if v.name == "Zernike_coef:0"][0]
        # tf.summary.image('deconvoled center layer', tf.expand_dims(test_result, axis=-1), max_outputs=3)

        all_variables = tf.trainable_variables()

        if isinstance(starter_learning_rate, dict): # this is for some special training strategy
            training_steps = []

            for key in starter_learning_rate: # with
                # Get the list of variables
                optimizer = self._set_up_optimizer(starter_learning_rate[key],
                                                    decay_type[key],
                                                    decay_params[key],
                                                    opt_type[key],
                                                    opt_params[key],
                                                    global_step=global_step)

                var_list = [var for var in all_variables if key in var.name]
                print("Single optimizer for following group of variables:")
                pprint(var_list)

                train_step = slim.learning.create_train_op(total_loss_graph,
                                                           optimizer=optimizer,
                                                           global_step=global_step,
                                                           variables_to_train=var_list, # trainable variable list
                                                           summarize_gradients=True)
                training_steps.append(train_step)

            train_step = tf.group(*training_steps)
        else:
            optimizer = self._set_up_optimizer(starter_learning_rate,
                                               decay_type,
                                               decay_params,
                                               opt_type,
                                               opt_params,
                                               global_step=global_step)
            train_step = slim.learning.create_train_op(total_loss_graph,
                                                       optimizer=optimizer,
                                                       variables_to_train=all_variables,
                                                       global_step=global_step,
                                                       summarize_gradients=True)

        # Attach summaries to salient training parameters
        # tf.summary.scalar('data_loss', data_loss_graph)
        # tf.summary.scalar('reg_loss', reg_loss_graph)
        # tf.summary.scalar('total_loss', total_loss_graph)

        # Create a saver for saving out the models, here by default we save everything
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2,
                max_to_keep=3)

        # Create saver to restore variables from checkpoints
        all_variables = set(slim.get_model_variables() + tf.trainable_variables())

        if self.ckpt_path is not None:
            if isinstance(self.ckpt_path, dict):
                restoration_savers = {}
                for var_id in self.ckpt_path:
                    variables = [var for var in all_variables if var_id in var.name]
                    print("Restoring the following variables:")
                    pprint([var.name for var in variables])
                    restoration_savers[var_id] = tf.train.Saver(variables)
            else:
                restoration_saver = tf.train.Saver()

        # Get all summaries
        # summaries_merged = tf.summary.merge_all()
        # summary_writer = tf.summary.FileWriter(logdir, self.sess.graph, flush_secs=60)

        # plug in a test deconvolution, added for THIS SPECIFIC PROJECT


        # Init op
        init = tf.global_variables_initializer()
        self.sess.run(init,feed_dict=feed_dict)

        if self.ckpt_path is not None:
            if isinstance(self.ckpt_path, dict):
                for var_id in self.ckpt_path:
                    print("Loading from checkpoint path %s"%self.ckpt_path)
                    restoration_savers[var_id].restore(self.sess, self.ckpt_path[var_id])
            else:
                restoration_saver.restore(self.sess, self.ckpt_path)

        # Train the model
        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)
        if self.tfdebug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

        tf.get_default_graph().finalize()
        print("Beginning the training")
        try:
            for step in range(num_steps):
                _, total_loss, reg_loss, data_loss1, data_loss2 = self.sess.run([train_step,
                                                                    total_loss_graph,
                                                                    reg_loss_graph,
                                                                    data_loss_graph1, 
                                                                    data_loss_graph2],
                                                                    feed_dict=feed_dict)
                print("Step %d   total_loss %0.8f   reg_loss %0.8f   data_loss1 %0.8f  data_loss2 %0.8f\n"%\
                        (step, total_loss, reg_loss, data_loss1, data_loss2))
                if coord.should_stop():
                    break

                if not step % num_steps_until_save and step:
                    print("Saving model...")
                    save_path = os.path.join(logdir, self.name+'.ckpt')

                    # diable the save for a very deep folder
                    # if decay_type is not None:
                        # self.saver.save(self.sess, save_path, global_step=global_step)
                    # else:
                        # self.saver.save(self.sess, save_path, global_step=step)

                if not step % num_steps_until_summary:
                    print("Writing summaries...")
                    # summary = self.sess.run(summaries_merged, feed_dict=feed_dict)
                    # summary_writer.add_summary(summary, step)

                    # save test deconvolution mat file, debug information
                    mat_test_result = self.sess.run(test_result, feed_dict=feed_dict) # size [1, height, width, channel]
                    mat_psf = self.sess.run(psfs, feed_dict=feed_dict) # size [1, height, width, channel]
                    mat_gamma = self.sess.run(gamma, feed_dict=feed_dict) # size [1, channel, 1, 1]
                    mat_x_test = self.sess.run(x_test, feed_dict=feed_dict) # size [1, height, width]
                    mat_phase = self.sess.run(var_phase, feed_dict=feed_dict) # size [height, width]
                    mat_phase = (mat_phase + np.pi) % (2 * np.pi)# wrap!
                    
                    mat_zernike_coef = self.sess.run(var_coef, feed_dict=feed_dict)

                    # permute the size order since in python they are all inversed
                    tifffile.imsave('{0}/test_deconv_{1:d}.tiff'.format(logdir, step), np.squeeze(mat_test_result.astype(dtype='float32'))) 
                    tifffile.imsave('{0}/psf_{1:d}.tiff'.format(logdir, step), np.squeeze(np.transpose(mat_psf, [3, 2, 1, 0]).astype(dtype='float32'))) 
                    tifffile.imsave('{0}/GT_{1:d}.tiff'.format(logdir, step), np.squeeze(mat_x_test.astype(dtype='float32'))) 
                    tifffile.imsave('{0}/phase_{1:d}.tiff'.format(logdir, step), np.squeeze(np.transpose(mat_phase, [1, 0]).astype(dtype='float32'))) 

                    # coef save
                    scipy.io.savemat('{0}/Zernike_coef_{1:d}.mat'.format(logdir, step),  mdict={'Zernike_from_tf': mat_zernike_coef})
                    scipy.io.savemat('{0}/deconv_gamma_{1:d}.mat'.format(logdir, step),  mdict={'gamma': mat_gamma})

            print("Saving final model.")
            save_path = os.path.join(logdir, self.name+'.ckpt')
        except Exception as e:
            print("Training interrupted due to exception")
            print(e)
            coord.request_stop()
        finally:
            coord.request_stop()
            coord.join(enqueue_threads)


        # add
        # if decay_type is not None:
        #     self.saver.save(self.sess, save_path, global_step=global_step)
        # else:
        #     self.saver.save(self.sess, save_path, global_step=step)
