import wandb
from absl import app
import tensorflow as tf
from src.datasets.common.dataset_skeleton import DatasetSkeleton
from src.util.cnn import get_model, make_ds_from
import logging
import traceback

# default parameters for evaluation
from src.util.json import DatasetDCICJson



from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean(name='soft', help='Use soft labels for training.', default=False)
flags.DEFINE_boolean(name='not_pretrained', help='Use no pretraining.', default=False)
flags.DEFINE_integer(name='epochs', default=50, help='Number of epochs')


wandb_usage = True

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



        # save guard against bugs
        try:
            # start logging for intermediate results
            # self.report.start_fold_anaylsis(f'{dataset_info.name}-{v_fold}', oracle.get_classes())


            # define variables
            batch_size = 32
            epochs = FLAGS.epochs
            lr = 1e-4

            dropout = 0.5
            network = 'resnet50v2'
            augmentation = {'prob_rotate': 0.5, 'prob_flip': 0.5, 'prob_color': 1, 'prob_zoom': 0.3,
                            'use_imgaug': False}

            self.num_classes = len(dataset_info.classes)


            model, val_ds, gt_val, all_ds, paths = self.train_loop(ds,dataset_info,batch_size,lr, epochs, dropout, network, augmentation,v_fold,num_annos,percentage_labeled,oracle)

            print("Predict & Evaluate for trained model")

            # save relabeled predicitons
            y_pred = model.predict(all_ds)

            for i, path in enumerate(paths):
                split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
                ds.update_image(path, split, [float(temp) for temp in y_pred[i, :self.num_classes]])

            y_pred = model.predict(val_ds)
            y_true = gt_val


            # Show evaluations
            self.report.end_run(ds, y_true, y_pred, verbose=2)

        except Exception as e:
            logging.error(traceback.format_exc())

        return ds

    def after_run_finished(self, ds : DatasetDCICJson):
        pass # done this already in the loop

    def train_loop(self, ds: DatasetDCICJson, dataset_info: DatasetSkeleton, batch_size,lr, epochs, dropout, network, augmentation, v_fold,num_annos,percentage_labeled,oracle):
        """"
        :param ds:
        :param dataset_info:
        :param batch_size:
        :param lr:
        :param epochs:
        :param dropout:
        :param network:
        :param augmentation:
        :param v_fold:
        :param num_annos:
        :param percentage_labeled:
        :param model:
        :return:
        """


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

        class_weights = dataset_info.class_weights

        print("Used class weights: %s" % class_weights)


        if wandb_usage:
            wandb_name = "#".join([self.name,dataset_name])

            config = {'batch_size':batch_size, 'lr':lr, 'epochs':epochs, 'dropout':dropout, 'network':network, 'dataset':dataset_info.name,
                      'fold':v_fold, 'num_annos':num_annos, 'percentage_labeled':percentage_labeled, 'num_classes':num_classes,
                      'num_train':len(paths_train), 'num_val':len(paths_val), 'num_all':len(paths),}

            run = wandb.init(project='proposal', name=wandb_name, config=config, tags=['benchmark'],
                             reinit=True)
        c = [wandb.keras.WandbCallback(save_model=False)] if wandb_usage else []


        loss = tf.keras.losses.categorical_crossentropy

        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=["accuracy"],
        )

        # train network
        model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, verbose=2, steps_per_epoch=steps_per_epoch,
                  class_weight=class_weights, callbacks=c)






        return model, val_ds, gt_val,  all_ds, paths

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
