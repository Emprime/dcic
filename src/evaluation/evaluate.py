import glob
import logging
import os

import itertools
import traceback
from os.path import join
from absl import app
from absl import flags
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from src.evaluation.report import DCICReport
from src.util.cnn import get_model, make_ds_from
from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson
from src.util.mixed import get_all_dataset_files
import wandb
from ray import tune
import tensorflow_addons as tfa
FLAGS = flags.FLAGS


flags.DEFINE_string(name='output_folder', default="/data/output_datasets",
                    help='The folder with the generated DCIC files which should be evaluated')

flags.DEFINE_list(name='folders',
                  help='the folders which should be evaluated', default=[])


flags.DEFINE_list(name='mode',
                  help='the type of input labels can be hard or soft. The loss is `Cross-entropy`for hard labels and `Kullback Leiber Divergence`for soft labels.',
                  default=['soft'])

flags.DEFINE_list(name='slices',
                  help='define the slices for the evaluation.',
                  default=[1, 2, 3])

flags.DEFINE_integer(name='verbose',
                  help='the verbosity setting determines, how much during the evaluation is shown (0 only final result, 1 results per folder, 2 results per slice, 3 results per slice + training logs.',
                  default=2)

flags.DEFINE_boolean(name='wandb', help="log to weights and biases", default=False)


flags.DEFINE_boolean(name='gap', help="Store intermedidate GAP Features for visualizations", default=False)


flags.DEFINE_boolean(name='provided_test', help="Indicates that the evaluation should be calculcated on the provided test data and not the original one", default=False)

def evaluation_function(config, dcicReport=None):
    """
    setup the training function for the experiment with the given config
    :return:
    """

    # entangle config
    # setup augmentations
    augs = [{'prob_rotate': 0, 'prob_flip': 0, 'prob_color': 0, 'prob_zoom': 0,
                    'use_imgaug': False},
             {'prob_rotate': 0.5, 'prob_flip': 0.5, 'prob_color': 1, 'prob_zoom': 0.3,
                    'use_imgaug': False},
             {'prob_rotate': 0.0, 'prob_flip': 0.0, 'prob_color': 0, 'prob_zoom': 0.0,
              'use_imgaug': True},
             {'prob_rotate': 0.5, 'prob_flip': 0.5, 'prob_color': 1, 'prob_zoom': 0.3,
              'use_imgaug': True}
             ]
    augmentation = augs[config['augmentation']]
    batch_size = config['batch_size']
    lr = config['lr']
    network = config['network']
    weights = config['weights']
    mode = config['mode']
    input_upsampling = config['input_upsampling']
    opt = config['opt']
    folder = config['folder']
    file = config['file']
    v_fold_index = config['v_fold_index']
    weight_decay = config['weight_decay']
    use_class_weights = config['use_class_weights']
    epochs = config['epochs']
    dropout = config['dropout']
    tuning = config['tuning']
    verbose = config['verbose']
    wandb_usage = config['wandb_usage']
    provided_test = config['provided_test']
    output_folder = config['output_folder']
    slices = config['slices']
    save_gap = config.get("gap",False)

    dcicReport = DCICReport() if dcicReport is None else dcicReport # create reporter if not given


    # setup wandb logging if desired
    try:
        if wandb_usage:
            wandb_name = f"{file[:8]}-bs{batch_size}-lr{lr}-{network}-aug{config['augmentation']}-{str(weights)}-m{mode}-i{input_upsampling}-{opt}"
            run = wandb.init(project='benchmark-eval', name=wandb_name, config=config, tags=['benchmark'],
                             reinit=True)
        if verbose > 0:
            print(f"###### START Experfiment {file} #######")
            print(f"Used config: {config}")




        # Train network
        dataset_json = DatasetDCICJson.from_file(join(output_folder, folder, file))
        dataset_name = dataset_json.dataset_name
        dataset_info = get_all_dataset_infos()[dataset_name]
        org_dataset_json = DatasetDCICJson.from_file(join(dataset_info.evaluate_directory,
                                                          "{name}-slice{split}.json".format(name=dataset_name,
                                                                                            split=v_fold_index + 1)))

        cl = dataset_json.classes
        num_classes = len(cl)
        num_items_not_test = len([1 for _, split, _, _ in dataset_json.get_image_iterator() if split != 'test'])


        if verbose > 1:
            print("found a dataset with %d images with %d non-test images and  %d classes" % (
                dataset_json.get_number_images(), num_items_not_test, num_classes))

        # create datasets based on the dataset json
        paths_train, gt_train = dataset_json.get_training_subsets('train',mode)
        input_ds_train, target_ds_train = make_ds_from(dataset_info,paths_train,gt_train, augmentation, True)

        paths_val, gt_val = dataset_json.get_training_subsets('val',mode)
        input_ds_val, target_ds_val = make_ds_from(dataset_info,paths_val,gt_val, augmentation, False)

        if provided_test:
            paths_test, gt_test = dataset_json.get_training_subsets('test', mode)
        else:
            paths_test, gt_test = org_dataset_json.get_training_subsets('test', mode)

        input_ds_test, target_ds_test = make_ds_from(dataset_info, paths_test, gt_test, augmentation, False)

        # prevent bug of to small sets
        batch_size = batch_size if batch_size < len(gt_train) else len(gt_train)


        train_ds = tf.data.Dataset.zip((
            input_ds_train, target_ds_train
        )).repeat().shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # shuffle before batching

        steps_per_epoch = (len(gt_train) // batch_size)

        val_ds = tf.data.Dataset.zip((
            input_ds_val, target_ds_val
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.zip((
            input_ds_test, target_ds_test
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)



        # setup model
        model, gap_model = get_model(dataset_name, num_classes, weights=weights, network_name=network, dropout=dropout, input_upsampling=dataset_info.input_sampling if input_upsampling else 0, get_gap_model=True)

        class_weights = dataset_info.class_weights

        if not use_class_weights:
            class_weights = None

        if verbose > 0:
            print("Used class weights: %s" % class_weights)
            print(f"Used {len(gt_train)} training samples and {len(gt_val)} validation samples and {len(gt_test)} test images")
            print(f"Example of used soft_gt for the first three samples: {gt_train[:3,:]}")

        loss = tf.keras.losses.categorical_crossentropy # if mode == 'hard' else tf.keras.losses.kl_divergence

        decay_steps = int(epochs * len(gt_train) / batch_size)

        if opt == "sgdwr":
            learning_rate_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(lr, first_decay_steps=decay_steps // 5)
        else:
            learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=decay_steps)

        step = tf.Variable(0, trainable=False)

        # lr and wd can be a function or a tensor
        learning_rate = lr * learning_rate_fn(step)
        wd = lambda: weight_decay * learning_rate_fn(step)

        if opt == "sgdw" or opt == "sgdwr":
            optimizer = tfa.optimizers.SGDW(
                learning_rate=learning_rate, weight_decay=wd, momentum=0.9)
        elif opt == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif opt == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        # train network
        if not save_gap:
            c = [wandb.keras.WandbCallback(save_model=False )] if wandb_usage else []
            model.fit(x=train_ds, validation_data=val_ds, epochs=epochs, verbose=2 if FLAGS.verbose == 3 else 0,
                      steps_per_epoch=steps_per_epoch,
                      class_weight=class_weights,
                      callbacks=c
                      )
        else:
            # special training store only for value extraction
            np.savetxt(join("/data/logs/gap_features/", f"{file}-gt.csv"), gt_test, delimiter=",")
            for ep in range(epochs):
                print("Manual Loop: " , ep)
                model.fit(x=train_ds, validation_data=val_ds, epochs=1, verbose=2 if FLAGS.verbose == 3 else 0,
                          steps_per_epoch=steps_per_epoch,
                          class_weight=class_weights
                          )

                features = gap_model.predict(test_ds)
                np.savetxt(join("/data/logs/gap_features/",f"{file}-{ep}.csv"), features, delimiter=",")

        if verbose > 1:
            print("Predict & Evaluate for trained model")

        # save outputs in specified format of sofia project
        y_pred = model.predict(test_ds)
        y_true = gt_test

        # Show evaluations
        f1, acc, kl = dcicReport.end_run(dataset_json, y_true, y_pred, verbose=verbose)

        if wandb_usage:
            wandb.log({'kl': kl, 'macro_f1': f1, 'macro_acc': acc})

        if tuning:
            # save elements to ray tune
            tune.report(kl=kl, macro_f1=f1,macro_acc=acc)

        if wandb_usage:
            run.finish()

    except Exception as e:
        print(e)
        traceback.print_exc()

        if tuning:
            # save elements to ray tune
            tune.report(kl=99, macro_f1=-1,macro_acc=-1)



def main(argv):
    print("Evaluation for DCIC Benchmark")

    dcicReport = DCICReport()

    slices = [int(i) for i in FLAGS.slices]

    try:
        for mode in FLAGS.mode:

            assert mode in ['hard', 'soft'], "Mode is not valid, can only be hard or soft"

            folders = FLAGS.folders

            # wildcard check
            if len(folders) == 1 and "*" in folders[0]:
                # print(os.listdir(FLAGS.output_folder))
                paths = glob.glob(join(FLAGS.output_folder,folders[0]))
                folders = sorted([path.split("/")[-1] for path in paths])
                print(f"Found the folders {folders} with pattern search")

            # for folder, org_dataset in zip(folders, orig_datasets):
            for folder in folders:
                print("Process folder %s with %s mode" % (folder, mode))

                # save guard against potentailly broken analysis
                try:

                    input_dataset_files = get_all_dataset_files(FLAGS.output_folder, folder)

                    num_slices = len(input_dataset_files)

                    if num_slices != len(FLAGS.slices):
                        print(f"WARNING: found {num_slices} files but specified {len(FLAGS.slices)}")
                        num_slices = len(FLAGS.slices)

                    assert num_slices <= 5, "Expects only up to 5 splits but received %d" % num_slices

                    # iterate over all inputs
                    for v_fold_index, f in enumerate(input_dataset_files):
                        # ensure index in slices
                        if (v_fold_index+1) not in slices:
                            print("Skip slice ", v_fold_index+1)
                            continue

                        # redudant load to get access to the info
                        dj = DatasetDCICJson.from_file(join(FLAGS.output_folder, folder, f))
                        dataset_name = dj.dataset_name
                        di = get_all_dataset_infos()[dataset_name]

                        # default parameters for evaluation
                        config = {'batch_size':di.hyperparameters['batch_size'], 'epochs':60, 'lr':di.hyperparameters['lr'], 'weights':di.hyperparameters['weights'],
                                  'dropout':di.hyperparameters['dropout'], 'network':di.hyperparameters['network'], 'augmentation':di.hyperparameters['augmentation'],
                                  'use_class_weights':True,'num_slices':num_slices, 'folder':folder,
                                  'mode':mode, 'file':f,  'opt': di.hyperparameters['opt'], 'input_upsampling':di.hyperparameters['input_upsampling'], 'weight_decay': di.hyperparameters['weight_decay'],
                                  'v_fold_index': v_fold_index, 'tuning':False, 'verbose': FLAGS.verbose, 'wandb_usage':FLAGS.wandb,
                                  'provided_test':FLAGS.provided_test, 'output_folder':FLAGS.output_folder, 'slices':FLAGS.slices,
                                  'gap':FLAGS.gap}




                        print(f"###### START Evaluation {f} #######")

                        evaluation_function(config,dcicReport)

                    # present but do not save for gap
                    dcicReport.summarize_and_reset(folder, mode, save=not FLAGS.gap, verbose=FLAGS.verbose)

                except Exception as e:
                    logging.error(traceback.format_exc())
    except Exception as e:
        pass
    finally:
        # print results regardless
        dcicReport.show()


if __name__ == '__main__':
    app.run(main)
