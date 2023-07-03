import glob
import logging
import traceback
from os.path import join
from absl import app
from absl import flags
from src.evaluation.report import DCICReport
from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson
from src.util.mixed import get_all_dataset_files
import torch.nn as nn
import numpy as np

import torch
from scipy.special import softmax
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, TrainingArguments, Trainer, \
    default_data_collator, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, load_metric, Features, ClassLabel, Array3D, Image, Sequence, Value, Dataset

from PIL import Image as pil_image

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


def make_ds(paths, gt, dataset_root,features):
    imgs = [pil_image.open(join(dataset_root, p)).convert('RGB') for p in paths]

    return Dataset.from_dict({'img': imgs, 'label': gt.astype(np.float32)}, features=features)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    return load_metric("accuracy").compute(predictions=predictions, references=labels)


class ViTForImageClassification2(nn.Module):
    def __init__(self, num_labels=10):
        super(ViTForImageClassification2, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0])

        # print(logits, labels)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss = loss_fct(torch.nn.functional.log_softmax(logits.view(-1, self.num_labels)),
                            labels.view(-1, self.num_labels))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
        paths_val, gt_val = dataset_json.get_training_subsets('val',mode)

        if provided_test:
            paths_test, gt_test = dataset_json.get_training_subsets('test', mode)
        else:
            paths_test, gt_test = org_dataset_json.get_training_subsets('test', mode)

        # prevent bug of to small sets
        batch_size = batch_size if batch_size < len(gt_train) else len(gt_train)


        # Make datasets
        # get into structure which is similar to others

        dataset_root = dataset_info.raw_data_root_directory

        features = Features({
            # 'label': ClassLabel(
            #     names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
            'label': Sequence(feature=Value(dtype="float32"), length=num_classes),
            'img': Image(decode=True, id=None),
        })

        train_ds = make_ds(paths_train, gt_train,dataset_root, features)
        val_ds = make_ds(paths_val, gt_val,dataset_root,features)
        test_ds = make_ds(paths_test, gt_test,dataset_root, features)


        # get models
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        data_collator = default_data_collator

        # preprocess images
        def preprocess_images(examples):
            images = examples['img']

            # load images

            # print(len(images))
            images = [np.array(image, dtype=np.uint8) for image in images]
            images = [np.moveaxis(image, source=-1, destination=0) for image in images]
            # print("before feature")
            inputs = feature_extractor(images=images)
            # print("after feature")
            examples['pixel_values'] = inputs['pixel_values']
            # examples['label'] = [[1 if e == i else 0 for i in range(10)] for e in examples['label']]

            # print(examples['label'])

            return examples

        features = Features({
            # 'label': ClassLabel(
            #     names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
            'label': Sequence(feature=Value(dtype="float32"), length=num_classes),
            'img': Image(decode=True, id=None),
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        })

        preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
        preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
        preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)

        # setup torch gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define vit training


        args = TrainingArguments(
            f"test-cifar-10",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=10,
            per_device_eval_batch_size=4,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            logging_dir='logs',
            save_strategy="no"
        )

        # model = ViTForImageClassification()
        model = ViTForImageClassification2(num_labels=num_classes)
        model.to(device)


        trainer = Trainer(
            model,
            args,
            train_dataset=preprocessed_train_ds,
            eval_dataset=preprocessed_val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )


        trainer.train()

        if verbose > 1:
            print("Predict & Evaluate for trained model")

        outputs = trainer.predict(preprocessed_test_ds)
        y = outputs.predictions


        y = softmax(y, axis=1)

        y_pred = y
        y_true = test_ds['label']
        # Show evaluations
        f1, acc, kl = dcicReport.end_run(dataset_json, y_true, y_pred, verbose=verbose)

        # print Scores as comparison
        print("KL DIV")

        print(dcicReport.kl_div(y_true, y_pred))

        epsilon = 1e-20
        y_true = np.clip(y_true, epsilon, 1 - epsilon)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        y_pred = np.log(y_pred)
        print(np.mean(np.sum(y_true * (np.log(y_true) - y_pred), axis=1)))

        c1 = torch.nn.KLDivLoss(reduction="mean")(torch.tensor(y_pred), torch.tensor(y_true))
        print("mean " + str(c1))

        c2 = torch.nn.KLDivLoss(reduction="batchmean")(torch.tensor(y_pred), torch.tensor(y_true))
        print("batchmean " + str(c2))


    except Exception as e:
        print(e)
        traceback.print_exc()

        # if tuning:
        #     # save elements to ray tune
        #     tune.report(kl=99, macro_f1=-1,macro_acc=-1)



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

                        # special for verse
                        if "verse" in f:
                            v_fold_index = 2  # will be added with +1 to 3

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

                    # add vit behind method name
                    splitted = folder.split("-")
                    method_split = splitted[1]
                    method_split += "_vit"
                    splitted[1] = method_split
                    folder = "-".join(splitted)


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
