import os
from os.path import join
from random import Random
import numpy as np
from absl import flags
import os
import cv2
from tqdm import tqdm
from urllib.request import urlopen
from tempfile import NamedTemporaryFile
from shutil import unpack_archive
from src.util.json import DatasetDCICJson
from src.util.oracle import AnnotationOracle

FLAGS = flags.FLAGS
flags.DEFINE_string(name='input_folder',
                    help='the main input folder for the input data to the benchmark pipeline (subfolder under the main data root directory)', default='input_datasets')

flags.DEFINE_string(name='evaluate_folder', default="evaluate_datasets",
                    help='The folder with a perfect ground-truth for the datasets (subfolder under the main data root directory')

hyper_parameters = {
 #   'Benthic': {'weights': 'imagenet', 'kl': '0.7382', 'macro_f1': '0.6572', 'macro_acc': '0.6487', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.0, 'network': 'resnet50v2_large', 'augmentation': 1, 'opt': 'sgdw', 'input_upsampling': False, 'weight_decay': 0.001},
 #   'CIFAR10H': {'weights': 'imagenet', 'kl': '0.4270', 'macro_f1': '0.9004', 'macro_acc': '0.9007', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.0, 'network': 'resnet50v2', 'augmentation': 0, 'opt': 'sgdwr', 'input_upsampling': True, 'weight_decay': 0.0005},
 #   'MiceBone': {'weights': 'imagenet', 'kl': '0.2927', 'macro_f1': '0.6469', 'macro_acc': '0.6607', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.0, 'network': 'resnet50v2_large', 'augmentation': 1, 'opt': 'sgdwr', 'input_upsampling': False, 'weight_decay': 0.001},
 #   'Pig' : {'weights': 'imagenet', 'kl': '0.4744', 'macro_f1': '0.4464', 'macro_acc': '0.4544', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.5, 'network':
 #       'resnet50v2_large', 'augmentation': 1, 'opt': 'sgdw', 'input_upsampling': False, 'weight_decay': 0.0005},
 #   'Plankton': {'weights': 'imagenet', 'kl': '0.2466', 'macro_f1': '0.8992', 'macro_acc': '0.9183', 'lr': 1e-05, 'batch_size': 128, 'dropout': 0.0, 'network': 'incepresv2', 'augmentation': 1, 'opt': 'adam', 'input_upsampling': True, 'weight_decay': 0.001},
 #   'QualityMRI': {'weights': 'imagenet', 'kl': '0.0625', 'macro_f1': '0.7723', 'macro_acc': '0.7580', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.5, 'network': 'densenet121', 'augmentation': 0, 'opt': 'sgdwr', 'input_upsampling': False, 'weight_decay': 0.001},
 #   'Synthetic': {'weights': 'imagenet', 'kl': '0.0638', 'macro_f1': '0.9048', 'macro_acc': '0.9058', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.5, 'network': 'densenet121', 'augmentation': 1, 'opt': 'sgdw', 'input_upsampling': True, 'weight_decay': 0.0005},
 #   'Treeversity#1': {'weights': 'imagenet', 'kl': '0.3876', 'macro_f1': '0.7972', 'macro_acc': '0.8044', 'lr': 1e-05, 'batch_size': 128, 'dropout': 0.0, 'network': 'densenet121', 'augmentation': 1, 'opt': 'adam', 'input_upsampling': True, 'weight_decay': 0.0005},
 #   'Treeversity#6': {'weights': 'imagenet', 'kl': '0.5853', 'macro_f1': '0.6636', 'macro_acc': '0.6763', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.5, 'network': 'resnet50v2_large', 'augmentation': 1, 'opt': 'sgdwr', 'input_upsampling': False, 'weight_decay': 0.001},
  #  'Turkey': {'weights': 'imagenet', 'kl': '0.2923', 'macro_f1': '0.7552', 'macro_acc': '0.7663', 'lr': 0.1, 'batch_size': 128, 'dropout': 0.5, 'network': 'resnet50v2_large', 'augmentation': 1, 'opt': 'sgdw', 'input_upsampling': False, 'weight_decay': 0.0005}
}




default_parameters = {
    'batch_size':32, 'lr':1e-4, 'weights':'imagenet', 'dropout':0.5, 'network':'resnet50v2', 'augmentation':1, 'opt': 'adam', 'input_upsampling':False, 'weight_decay': 0.0005,
}

class DatasetSkeleton:
    """
    Skeleton class for all datasets in the benchmark
    """

    def __init__(self, name, input_size, classes, number_images_per_fold, input_sampling, number_images_per_class,channel_mean, channel_std, root_directory):
        self.name = name
        self.input_size = input_size
        self.classes = classes
        self.num_classes = len(self.classes)
        self.number_images_per_fold = number_images_per_fold
        self.root_directory = root_directory
        self.input_sampling = input_sampling
        self.number_images_per_class = np.array(number_images_per_class)#
        # based on the formula n_samples / (n_classes * np.bincount(y)) defined at
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        self.class_weights = dict(zip(
            np.arange(self.num_classes),
            np.sum(self.number_images_per_class) / (self.num_classes * self.number_images_per_class)
                    ))

        # RGB
        self.channel_mean = channel_mean
        self.channel_std = channel_std

        # setup paths

        raw_folder = "raw_datasets"
        self.raw_data_directory = join(self.root_directory,raw_folder, self.name)
        self.raw_data_root_directory = join(self.root_directory,raw_folder)
        self.evaluate_directory = join(self.root_directory, FLAGS.evaluate_folder, self.name)
        self.input_directory = join(self.root_directory, FLAGS.input_folder, self.name)

        # sanity checks
        assert self.num_classes > 1, "Need to provide at least two classes"
        assert len(number_images_per_fold) == 5, "Expected images in 5 folds"
        assert self.input_size > 1, "Need at least one positive pixel for a classification"
        assert os.path.exists(root_directory), "The provided root directory does not exist"

        # setup hyperparameters
        self.hyperparameters = hyper_parameters.get(self.name, default_parameters)

    def check_setup(self):
        """
        check if the dataset is correctly prepared for the usage in the benchmark.
        do the necessary setup if possible.
        """

        # check if data is available and check number of images
        directory = self.raw_data_directory

        if not os.path.exists(directory):
            print(f"Download raw files to {directory}, This can take a moment ...")

            zipurl = f'https://zenodo.org/record/8115942/files/{self.name.replace("#","")}.zip'
            with urlopen(zipurl) as zipresp, NamedTemporaryFile() as tfile:
                tfile.write(zipresp.read())
                tfile.seek(0)
                unpack_archive(tfile.name, self.raw_data_root_directory, format='zip')

            print(f"Successfully downloaded and extrated raw data for {self.name}")

        assert os.path.exists(directory), f"No raw files found under {directory}"
        for i in range(5):
            files = os.listdir(join(directory, f"fold{i+1}"))
            assert len(files) == self.number_images_per_fold[i], f"Wrong images given in fold {i+1}, found {len(files)}"

        # check if annotations are available
        assert os.path.exists(join(directory,"annotations.json")), f"No Annotations (annotations.json) available under {directory}"

        # check if dataset slices are available
        if not os.path.exists(self.input_directory):
            # try to create the slices
            self.generated_slices(self.input_directory)

        assert os.path.exists(self.input_directory), f"Directory ({self.input_directory}) for slices is missing, Creation attempted failed"
        assert len(os.listdir(self.input_directory)) == 5, f"Could not find the 5 generated slices files inside {self.input_directory}, Creation attempted failed, Maybe you have an empty file left from a previous execution"


        # check if evaluation data is available

        if not os.path.exists(self.evaluate_directory):
            # try to create the evaluation slices
            self.generated_slices(self.evaluate_directory, use_gt=True)

        assert os.path.exists(
            self.evaluate_directory), f"Evaluation directory ({self.evaluate_directory})is missing, Creation attempted failed"
        assert len(os.listdir(
            self.evaluate_directory)) == 5, f"Could not find the 5 generated evaluation files inside {self.evaluate_directory}, Creation attempted failed.  Maybe you have an empty file left from a previous execution"

        # calculate and compare the mean and std of the channels
        ds = DatasetDCICJson.from_file(join(self.evaluate_directory,f'{self.name}-slice{1}.json'))
        paths, targets = ds.get_training_subsets('all', also_gt=True)
        targets = np.argmax(targets,axis=1)

        class_indices, counts = np.unique(targets,return_counts=True)

        for cl_index, c in zip(class_indices,counts):
            assert c == self.number_images_per_class[cl_index], f"Mismatch between specified and found images per class," \
                f" found {c}, specified{self.number_images_per_class[cl_index]}"

        channel_sum = np.zeros((3,))
        channel_squared_sum = np.zeros((3,))
        for p in tqdm(paths):
            img = cv2.imread(join(self.raw_data_root_directory,p), cv2.IMREAD_COLOR) # bgr image
            img = img.astype(float) / 255.
            channel_sum += np.mean(img, axis=(0,1))
            channel_squared_sum += np.mean(img**2, axis=(0,1))

        mean = channel_sum / len(paths)
        std = (channel_squared_sum / len(paths) - mean ** 2) ** 0.5

        # BGR to RGB
        mean = [mean[i] for i in [2,1,0]]
        std = [std[i] for i in [2,1,0]]

        # check
        print("Calculated Channel mean and std", mean,std)
        for i in range(2):
            assert abs(mean[i] - self.channel_mean[i]) < 0.001
            assert abs(std[i] - self.channel_std[i]) < 0.001


    def generated_slices(self, directory, use_gt=False):
        """
        generate the slices files for the given directory
        :param use_gt: use the oracle gt for the label annotation
        :return:
        """
        # setup variables and ensure structure
        os.makedirs(directory, exist_ok=True)
        oracle = AnnotationOracle(join(self.raw_data_directory, 'annotations.json'))

        assert oracle.classes == self.classes, f"Mismatch between classes {oracle.classes} vs. {self.classes}"

        print(f"# Generate {'Evaluation-' if use_gt else ''}Slice for {self.name}")


        for v_fold in [1,2,3,4,5]:

            print("## Create slice with validation fold %d" % v_fold)

            # determine train, val and test folds
            fold_assignments = ['train', 'train', 'train', 'train', 'train']
            fold_assignments[(v_fold + 1) % 5] = 'test'
            fold_assignments[(v_fold) % 5] = 'val'

            # init dataset
            datasetDCIC = DatasetDCICJson.from_definition(self.name, v_fold, self.classes)
            oracle.init(datasetDCIC)

            # generate warning for imbalanced folds
            class_distributions = np.zeros((5,len(self.classes)))

            # iterate over folds
            for i in range(5):
                print("### Preprocess fold %d" % i)
                fold_folder = join(self.raw_data_directory, "fold%d" % (i + 1))

                files = os.listdir(fold_folder)
                files = sorted(files)

                for j, file in tqdm(enumerate(files)):

                    file_name = join(self.name, "fold%d" % (i + 1), file)
                    if use_gt:
                        soft_gt = list(oracle.get_soft_gt(file_name, -1))
                        class_distributions[i,np.argmax(soft_gt)] += 1
                    else:
                        soft_gt = []
                    datasetDCIC.add_image(file_name, fold_assignments[i], soft_gt)

                    if j < 10:
                      print(file_name,soft_gt)

            # normalize distributions
            if np.sum(class_distributions) > 0 and v_fold == 1: # report only if entries are available and only once
                # print(class_distributions / np.sum(class_distributions, axis=1, keepdims=True))
                class_distributions /= np.sum(class_distributions, axis=1, keepdims=True)
                print(class_distributions)
                if np.any(np.abs(class_distributions-np.mean(class_distributions,axis=0,keepdims=True)) > 0.05):
                    print(f"WARNING: Relative distributions between folds is skewed! More than 5% deviation from the mean per classs. "
                          f"Details (row folds, columns averaged elements per fold)\n{class_distributions}")
                # oracle.annoJson.print_statistics()

            dcic_name = f'{self.name}-slice{v_fold}.json'
            datasetDCIC.save_jsons(join(directory, dcic_name))


    def get_number_of_images_per_fold(self):
        """
        get an array of number of images for the 5 folds
        :return:
        """
        return self.number_images_per_fold


    def get_input_size(self):
        """
        get the input size (width, height) for this dataset
        :return:
        """
        return self.input_size
