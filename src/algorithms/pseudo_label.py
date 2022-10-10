from os.path import join
from absl import app
from absl import flags
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from src.evaluation.report import DCICReport
from src.util.cnn import get_model, make_ds_from
import logging
import traceback

# default parameters for evaluation
from src.util.json import DatasetDCICJson

batch_size = 32
epochs = 50
lr = 1e-4

dropout = 0.5
network = 'resnet50v2'
augmentation = {'prob_rotate': 0.5, 'prob_flip': 0.5, 'prob_color': 1, 'prob_zoom': 0.3,
                                'use_imgaug': False}

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean(name='soft', help='Use soft labels for training.', default=False)
flags.DEFINE_boolean(name='not_pretrained', help='Use no pretraining.', default=False)

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton


class Pseudo_label(AlgorithmSkelton):

    def __init__(self):
        name = 'pseudo' if not FLAGS.soft else 'pseudo_soft'
        if FLAGS.not_pretrained:
            name += "_no_pretrain"
        AlgorithmSkelton.__init__(self,name)
        self.num_classes = 0
        self.mode  = 'hard' if not FLAGS.soft else 'soft'

        self.weights = 'imagenet' if not FLAGS.not_pretrained else None

    def run(self, ds , oracle, dataset_info, v_fold,num_annos,percentage_labeled):
        global batch_size

        # save guard against bugs
        try:
            # start logging for intermediate results
            # self.report.start_fold_anaylsis(f'{dataset_info.name}-{v_fold}', oracle.get_classes())

            # init values
            dataset_name = dataset_info.name
            self.num_classes = len(dataset_info.classes)
            num_classes = self.num_classes

            # create datasets based on the dataset json
            paths_train, gt_train = ds.get_training_subsets('train', self.mode)
            input_ds_train, target_ds_train = make_ds_from(dataset_info, paths_train, gt_train, augmentation, True)

            paths_val, gt_val = ds.get_training_subsets('val', self.mode)
            input_ds_val, target_ds_val = make_ds_from(dataset_info, paths_val, gt_val, augmentation, False)

            paths, targets = ds.get_training_subsets('all', self.mode)
            input_ds_all, _ = make_ds_from(dataset_info, paths, targets, augmentation, False)

            # prevent bug of to small sets
            batch_size = batch_size if batch_size < len(gt_train) else len(gt_train)

            # setup datasets
            train_ds = tf.data.Dataset.zip((
                input_ds_train, target_ds_train
            )).repeat().shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # shuffle before batching

            val_ds = tf.data.Dataset.zip((
                input_ds_val, target_ds_val
            )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

            all_ds = input_ds_all.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            steps_per_epoch = (len(gt_train) // batch_size)

            # setup model

            model = get_model(dataset_name, num_classes, weights=self.weights, network_name=network, dropout=dropout)

            # model.summary()


            class_weights = dataset_info.class_weights

            print("Used class weights: %s" % class_weights)

            loss = tf.keras.losses.categorical_crossentropy

            model.compile(
                loss=loss,
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                metrics=["accuracy"],
            )

            # train network
            model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch,
                      class_weight=class_weights)

            print("Predict & Evaluate for trained model")


            # save relabeled predicitons
            y_pred = model.predict(all_ds)

            for i, path in enumerate(paths):
                split = ds.get(path,'original_split')  # determine original split before move to unlabeled
                ds.update_image(path, split, [float(temp) for temp in y_pred[i, :]])


            # save outputs
            y_pred = model.predict(val_ds)
            y_true = gt_val


            # Show evaluations
            self.report.end_run(ds, y_true, y_pred, verbose=2)

        except Exception as e:
            logging.error(traceback.format_exc())

        return ds

    def after_run_finished(self, ds : DatasetDCICJson):
        pass # done this already in the loop

def main(argv):
    """
       Apply only initial annotation
       :return:
    """


    alg = Pseudo_label()
    alg.apply_algorithm()

    alg.report.show()

if __name__ == '__main__':
    app.run(main)
