# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResNet-50 on ImageNet using SNGP.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to hidden weights, and then replace the dense output layer with
a Gaussian process.

## Note:

Different from the paper, this implementation computes the posterior using the
Laplace approximation based on the Gaussian likelihood (i.e., squared loss)
rather than that based on cross-entropy loss. As a result, the logits for all
classes share the same covariance. In the experiments, this approach is shown to
perform better and computationally more scalable when the number of output
classes are large.

## References:
[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
[3]: Felix Xinnan Yu et al. Orthogonal Random Features. In _Neural Information
     Processing Systems_, 2016.
     https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf
"""

# copied from https://raw.githubusercontent.com/google/uncertainty-baselines/main/baselines/imagenet/sngp.py
from __future__ import print_function
import os
from os.path import join

import time

from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import robustness_metrics as rm
import tensorflow as tf
# import tensorflow_datasets as tfds
# import uncertainty_baselines as ub
# import utils  # local file import from baselines.imagenet
from tensorflow.keras.applications import imagenet_utils

from src.algorithms.sgnp_model import resnet50_sngp

import os
from tqdm import tqdm
import numpy as np
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from absl import app
from absl import flags
# IMPLEMENTATION of ELR Plus is based on the official repo https://github.com/shengliu66/ELR with the clothing1m Dataset
from src.algorithms.sgnp_scheduler import WarmUpPiecewiseConstantSchedule
from src.util.cnn import get_parsing_function

FLAGS = flags.FLAGS

# SGNP Sepcific
flags.DEFINE_integer('per_core_batch_size', 32, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
# flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 90, 'Number of training epochs.')
flags.DEFINE_integer('corruptions_interval', 90,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer(
    'checkpoint_interval', -1,
    'Number of epochs between saving checkpoints. Use -1 to '
    'only save the last checkpoints.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')
flags.DEFINE_float('train_proportion', default=1.0,
                   help='only use a proportion of training set and use the'
                        'rest for validation instead of the test set.')
flags.register_validator('train_proportion',
                         lambda tp: tp > 0.0 and tp <= 1.0,
                         message='--train_proportion must be in (0, 1].')

# Dropout flags.
flags.DEFINE_bool('use_mc_dropout', False,
                  'Whether to use Monte Carlo dropout during inference.')
flags.DEFINE_float('dropout_rate', 0., 'Dropout rate.')
flags.DEFINE_bool(
    'filterwise_dropout', True, 'Dropout whole convolutional'
                                'filters instead of individual values in the feature map.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of samples to use for MC Dropout prediction.')

# Spectral normalization flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', 6.,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 1.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', False,
    'Whether to normalize the input for GP layer using LayerNorm. This is '
    'similar to applying automatic relevance determination (ARD) in the '
    'classic GP literature.')
flags.DEFINE_string(
    'gp_random_feature_type', 'orf',
    'The type of random feature to use. One of "rff" (random Fourier feature), '
    '"orf" (orthogonal random feature) [3].')
flags.DEFINE_float('gp_cov_ridge_penalty', 1.,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', -1.,
    'The discount factor to compute the moving average of precision matrix.'
    'If -1 then instead compute the exact covariance at the lastest epoch.')
flags.DEFINE_float(
    'gp_mean_field_factor', 1.,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')
flags.DEFINE_bool(
    'gp_output_imagenet_initializer', True,
    'Whether to initialize GP output layer using Gaussian with small '
    'standard deviation (sd=0.01).')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
# TODO(jereliu): Support use_bfloat16=True which currently raises error with
# spectral normalization.
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS


# # Number of images in ImageNet-1k train dataset.
# APPROX_IMAGENET_TRAIN_IMAGES = int(1281167 * FLAGS.train_proportion)
#
# NUM_CLASSES = 1000

class SGNP(AlgorithmSkelton):
    def __init__(self):
        AlgorithmSkelton.__init__(self, 'sgnp')

    def parse_image(self, filename):
        """
        parse the image but apply no preprocessing
        :param filename:
        :return:
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        return image

    def inputs_to_dataset(self, dataset_info, paths, targets):
        # apply preprocessing

        dataset_root, dataset_name = dataset_info.raw_data_root_directory, dataset_info.name

        path_full = [join(dataset_root, path) for path in paths]

        list_ds = tf.data.Dataset.from_tensor_slices(path_full)
        if len(paths) > 0:
            images_ds = list_ds.map(self.parse_image)

            images_ds = images_ds.map(lambda x: imagenet_utils.preprocess_input(x, mode='tf'),
                                      num_parallel_calls=tf.data.AUTOTUNE)
        else:
            images_ds = list_ds  # no images loadable

        # print(soft_gt)

        labels_ds = tf.data.Dataset.from_tensor_slices(targets)

        dataset = tf.data.Dataset.zip((
            images_ds, labels_ds
        ))
        dataset = dataset.map(lambda x, y: {'features': x, 'labels': y})

        return dataset

    def run(self, ds, oracle, dataset_info, v_fold, num_annos, percentage_labeled):

        NUM_CLASSES = dataset_info.num_classes

        tf.io.gfile.makedirs(FLAGS.output_dir)
        logging.info('Saving checkpoints at %s', FLAGS.output_dir)
        tf.random.set_seed(FLAGS.seed)

        batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores


        # own datasets
        mode = 'hard'
        paths_train, gt_train = ds.get_training_subsets('train', mode)
        gt_train = np.argmax(gt_train, axis=1)
        train_ds = self.inputs_to_dataset(dataset_info, paths_train, gt_train)

        paths_val, gt_val = ds.get_training_subsets('val', mode)
        gt_val = np.argmax(gt_val, axis=1)
        val_ds = self.inputs_to_dataset(dataset_info, paths_val, gt_val)

        # prevent bug of to small sets
        batch_size = batch_size if batch_size < len(gt_train) else len(gt_train)

        # setup datasets
        train_ds = train_ds.repeat().shuffle(100).batch(batch_size).prefetch(
            tf.data.AUTOTUNE)  # shuffle before batching

        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # all_ds = input_ds_all.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = (len(gt_train) // batch_size)
        steps_per_eval = (len(gt_val) // batch_size)



        train_dataset = train_ds
        clean_test_dataset = val_ds

        test_datasets = {
            'clean': clean_test_dataset
        }


        if FLAGS.use_bfloat16:
            tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

        # input size
        w_h = 224  # dataset_info.get_input_size()

        # with strategy.scope():
        logging.info('Building Keras ResNet-50 model')
        model = resnet50_sngp(
            input_shape=(w_h, w_h, 3),
            batch_size=None,
            num_classes=NUM_CLASSES,
            use_mc_dropout=FLAGS.use_mc_dropout,
            dropout_rate=FLAGS.dropout_rate,
            filterwise_dropout=FLAGS.filterwise_dropout,
            use_gp_layer=FLAGS.use_gp_layer,
            gp_hidden_dim=FLAGS.gp_hidden_dim,
            gp_scale=FLAGS.gp_scale,
            gp_bias=FLAGS.gp_bias,
            gp_input_normalization=FLAGS.gp_input_normalization,
            gp_random_feature_type=FLAGS.gp_random_feature_type,
            gp_cov_discount_factor=FLAGS.gp_cov_discount_factor,
            gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty,
            gp_output_imagenet_initializer=FLAGS.gp_output_imagenet_initializer,
            use_spec_norm=FLAGS.use_spec_norm,
            spec_norm_iteration=FLAGS.spec_norm_iteration,
            spec_norm_bound=FLAGS.spec_norm_bound)
        logging.info('Model input shape: %s', model.input_shape)
        logging.info('Model output shape: %s', model.output_shape)
        logging.info('Model number of weights: %s', model.count_params())
        # Scale learning rate and decay epochs by vanilla settings.
        base_lr = FLAGS.base_learning_rate * batch_size / 256
        decay_epochs = [
            (FLAGS.train_epochs * 30) // 90,
            (FLAGS.train_epochs * 60) // 90,
            (FLAGS.train_epochs * 80) // 90,
        ]
        learning_rate = WarmUpPiecewiseConstantSchedule(
            steps_per_epoch=steps_per_epoch,
            base_learning_rate=base_lr,
            decay_ratio=0.1,
            decay_epochs=decay_epochs,
            warmup_epochs=5)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=1.0 - FLAGS.one_minus_momentum,
                                            nesterov=True)
        metrics = {
            'train/negative_log_likelihood': tf.keras.metrics.Mean(),
            'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'train/loss': tf.keras.metrics.Mean(),
            'train/ece': rm.metrics.ExpectedCalibrationError(
                num_bins=FLAGS.num_bins),
            'test/negative_log_likelihood': tf.keras.metrics.Mean(),
            'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
            'test/ece': rm.metrics.ExpectedCalibrationError(
                num_bins=FLAGS.num_bins),
            'test/stddev': tf.keras.metrics.Mean(),
        }


        logging.info('Finished building Keras ResNet-50 model')

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        initial_epoch = 0


        summary_writer = tf.summary.create_file_writer(
            os.path.join(FLAGS.output_dir, 'summaries'))

        @tf.function
        def train_step(iterator):
            """Training StepFn."""

            def step_fn(inputs, step):
                """Per-Replica StepFn."""
                images = inputs['features']
                labels = inputs['labels']

                if tf.equal(step, 0) and FLAGS.gp_cov_discount_factor < 0:
                    # Reset covaraince estimator at the begining of a new epoch.
                    if FLAGS.use_gp_layer:
                        model.layers[-1].reset_covariance_matrix()

                with tf.GradientTape() as tape:
                    logits = model(images, training=True)

                    if isinstance(logits, (list, tuple)):
                        # If model returns a tuple of (logits, covmat), extract logits
                        logits, _ = logits
                    if FLAGS.use_bfloat16:
                        logits = tf.cast(logits, tf.float32)

                    negative_log_likelihood = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                        logits,
                                                                        from_logits=True))
                    filtered_variables = []
                    for var in model.trainable_variables:
                        # Apply l2 on the weights. This excludes BN parameters and biases, but
                        # pay caution to their naming scheme.
                        if 'kernel' in var.name or 'bias' in var.name:
                            filtered_variables.append(tf.reshape(var, (-1,)))

                    l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
                        tf.concat(filtered_variables, axis=0))
                    # Scale the loss given the TPUStrategy will reduce sum all gradients.
                    loss = negative_log_likelihood + l2_loss
                    scaled_loss = loss / 1  # strategy.num_replicas_in_sync

                grads = tape.gradient(scaled_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                probs = tf.nn.softmax(logits)
                metrics['train/ece'].add_batch(probs, label=labels)
                metrics['train/loss'].update_state(loss)
                metrics['train/negative_log_likelihood'].update_state(
                    negative_log_likelihood)
                metrics['train/accuracy'].update_state(labels, logits)

            for step in tf.range(tf.cast(steps_per_epoch, tf.int32)):
                step_fn(next(iterator), step)
            # strategy.run(step_fn, args=(next(iterator), step))

        @tf.function
        def test_step(iterator, dataset_name):
            """Evaluation StepFn."""

            def step_fn(inputs):
                """Per-Replica StepFn."""
                images = inputs['features']
                labels = inputs['labels']

                logits_list = []
                stddev_list = []
                for _ in range(FLAGS.num_dropout_samples):
                    logits = model(images, training=False)

                    if isinstance(logits, (list, tuple)):
                        # If model returns a tuple of (logits, covmat), extract both
                        logits, covmat = logits
                    else:
                        covmat = tf.eye(FLAGS.per_core_batch_size)

                    if FLAGS.use_bfloat16:
                        logits = tf.cast(logits, tf.float32)

                    logits = ed.layers.utils.mean_field_logits(
                        logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
                    stddev = tf.sqrt(tf.linalg.diag_part(covmat))

                    stddev_list.append(stddev)
                    logits_list.append(logits)

                # Logits dimension is (num_samples, batch_size, num_classes).
                logits_list = tf.stack(logits_list, axis=0)
                stddev_list = tf.stack(stddev_list, axis=0)

                stddev = tf.reduce_mean(stddev_list, axis=0)
                probs_list = tf.nn.softmax(logits_list)
                probs = tf.reduce_mean(probs_list, axis=0)

                labels_broadcasted = tf.broadcast_to(
                    labels, [FLAGS.num_dropout_samples, tf.shape(labels)[0]])
                log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
                    labels_broadcasted, logits_list, from_logits=True)
                negative_log_likelihood = tf.reduce_mean(
                    -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
                    tf.math.log(float(FLAGS.num_dropout_samples)))

                if dataset_name == 'clean':
                    metrics['test/negative_log_likelihood'].update_state(
                        negative_log_likelihood)
                    metrics['test/accuracy'].update_state(labels, probs)
                    metrics['test/ece'].add_batch(probs, label=labels)
                    metrics['test/stddev'].update_state(stddev)


            for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
                step_fn(next(iterator))
            # strategy.run(step_fn, args=(next(iterator),))

        metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

        train_iterator = iter(train_dataset)
        start_time = time.time()

        for epoch in range(initial_epoch, FLAGS.train_epochs):
            logging.info('Starting to run epoch: %s', epoch)
            train_step(train_iterator)

            current_step = (epoch + 1) * steps_per_epoch
            max_steps = steps_per_epoch * FLAGS.train_epochs
            time_elapsed = time.time() - start_time
            steps_per_sec = float(current_step) / time_elapsed
            eta_seconds = (max_steps - current_step) / steps_per_sec
            message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                       'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                current_step / max_steps,
                epoch + 1,
                FLAGS.train_epochs,
                steps_per_sec,
                eta_seconds / 60,
                time_elapsed / 60))
            logging.info(message)

            datasets_to_evaluate = {'clean': test_datasets['clean']}
            if (FLAGS.corruptions_interval > 0 and
                    (epoch + 1) % FLAGS.corruptions_interval == 0):
                datasets_to_evaluate = test_datasets
            for dataset_type, test_dataset in datasets_to_evaluate.items():
                test_iterator = iter(test_dataset)
                logging.info('Testing on dataset %s', dataset_type)
                logging.info('Starting to run eval at epoch: %s', epoch)
                test_start_time = time.time()
                test_step(test_iterator, dataset_type)
                ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
                metrics['test/ms_per_example'].update_state(ms_per_example)

                logging.info('Done with testing on %s', dataset_type)

            corrupt_results = {}


            logging.info('Train Loss: %.4f, Accuracy: %.2f%%, ECE:  %.2f%%',
                         metrics['train/loss'].result(),
                         metrics['train/accuracy'].result() * 100,
                         metrics['train/ece'].result()['ece'] * 100)
            logging.info('Test NLL: %.4f, Accuracy: %.2f%%, ECE:  %.2f%%',
                         metrics['test/negative_log_likelihood'].result(),
                         metrics['test/accuracy'].result() * 100,
                         metrics['test/ece'].result()['ece'] * 100)
            total_results = {name: metric.result() for name, metric in metrics.items()}
            total_results.update(corrupt_results)
            # Metrics from Robustness Metrics (like ECE) will return a dict with a
            # single key/value, instead of a scalar.
            total_results = {
                k: (list(v.values())[0] if isinstance(v, dict) else v)
                for k, v in total_results.items()
            }
            with summary_writer.as_default():
                for name, result in total_results.items():
                    tf.summary.scalar(name, result, step=epoch + 1)

            for metric in metrics.values():
                metric.reset_states()

            if (FLAGS.checkpoint_interval > 0 and
                    (epoch + 1) % FLAGS.checkpoint_interval == 0):
                checkpoint_name = checkpoint.save(os.path.join(
                    FLAGS.output_dir, 'checkpoint'))
                logging.info('Saved checkpoint to %s', checkpoint_name)

        # Save final checkpoint.
        final_checkpoint_name = checkpoint.save(
            os.path.join(FLAGS.output_dir, 'checkpoint'))
        logging.info('Saved last checkpoint to %s', final_checkpoint_name)

        # Export final model as SavedModel.
        final_save_name = os.path.join(FLAGS.output_dir, 'model')
        model.save(final_save_name)
        logging.info('Saved model to %s', final_save_name)

        # predict all images with model
        paths, _ = ds.get_training_subsets('all', mode)
        targets = np.zeros((len(paths)))  # fake tarbdg input
        dataset = self.inputs_to_dataset(dataset_info, paths, targets)

        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_predict = (len(paths) // batch_size)

        def predict_step(iterator):
            """Evaluation StepFn."""

            def step_fn(inputs):
                """Per-Replica StepFn."""
                images = inputs['features']
                logits = model(images, training=False)

                if isinstance(logits, (list, tuple)):
                    # If model returns a tuple of (logits, covmat), extract both
                    logits, covmat = logits
                else:
                    covmat = tf.eye(FLAGS.per_core_batch_size)

                if FLAGS.use_bfloat16:
                    logits = tf.cast(logits, tf.float32)

                logits = ed.layers.utils.mean_field_logits(
                    logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)

                probs = tf.nn.softmax(logits).numpy()


                predictions.extend([probs[i, :] for i in range(len(probs))])

            predictions = []
            for _ in tf.range(tf.cast(steps_predict + 1, tf.int32)):
                step_fn(next(iterator))

            return predictions

        all_iterator = iter(dataset)
        dataset_name = dataset_info.name
        logging.info('Predict all data on dataset %s', dataset_name)
        predictions = predict_step(all_iterator)

        print(f"Predicted for {len(predictions)} images and found {len(paths)} images")
        for i, path in enumerate(paths):
            split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
            ds.update_image(path, split, [float(temp) for temp in predictions[i]])

        return ds


def main(argv):
    alg = SGNP()
    alg.apply_algorithm()


if __name__ == '__main__':
    app.run(main)
