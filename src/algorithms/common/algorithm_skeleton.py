import traceback
from os.path import join
from random import Random
import os
from absl import app
from absl import flags
import numpy as np

from src.datasets.common.dataset_skeleton import DatasetSkeleton
from src.evaluation.report import DCICReport
from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson
from src.util.mixed import GetDatasetNames
from src.util.oracle import AnnotationOracle

FLAGS = flags.FLAGS

flags.DEFINE_integer(name='number_annotations_per_image', default=-1, help='The number of annotations which are used to calculate the label of this dataset.'
                                                                          'The value -1 will be interpreted as an interation of all default values. The value of percentage_labeled_data has also to be -1 in this case. A value of -2 uses only all SSL settings.')

flags.DEFINE_float(name='percentage_labeled_data', default=-1,
                   help='Determines the amount of annotated data in relation to the complete data. '
                        'The value -1 will be interpreted as an interation of all default values. The value of number_annotations_per_image has also to be -1 in this case. A value of -2 uses only all SSL settings.')

flags.DEFINE_string(name='output_folder', default="output_datasets",
                    help='The folder with the generated DCIC files which should be evaluated, it es recommend to leave this value on the default value')

flags.DEFINE_list(name='datasets',
                    help='the dataset_names which should be processed, all is a special name which is replaced with all dataset names', default=['all'])

flags.DEFINE_integer(name='v_fold', default=-1,
                     help=' defines on which slice / which validation fold the algorithm is applied / uses. Can be 1 to 5 or -1 (all slices)')





class AlgorithmSkelton:
    def __init__(self, name, v_fold=None, dataset_name = None ):
        assert "-" not in name, "The character '-' is not allowd in an algorithm name for formatting reasons"
        self.name = name

        if v_fold is None:
            v_fold = FLAGS.v_fold

        if dataset_name is None:
            dataset_name = FLAGS.datasets
            # get all corresponding folder names for the datasets
            self.folder_dataset_names = GetDatasetNames(dataset_name)
        else:
            self.folder_dataset_names = [dataset_name]

        if FLAGS.percentage_labeled_data == -1 or FLAGS.number_annotations_per_image == -1:
            # special cases
            assert FLAGS.percentage_labeled_data == -1, 'A value of -1 indicates the special case of the default values for the initialization, Both "number_annotations_per_image" and "percentage_labeled_data" have to be -1 in this case.'
            assert FLAGS.number_annotations_per_image == -1, 'A value of -1 indicates the special case of the default values for the initialization, Both "number_annotations_per_image" and "percentage_labeled_data" have to be -1 in this case.'

            #  percentage, number
            self.init_values = [
                (0.1,1),(0.2,1),(0.5,1),(1,1), # semi supervised
                (1,3),(1,5),(1,10), # multiple
                (0.1,10),(0.2,5),(0.5,2) # semi-supervised + multiple
            ]
        elif FLAGS.percentage_labeled_data == -2 or FLAGS.number_annotations_per_image == -2:
            # special cases
            assert FLAGS.percentage_labeled_data == -2, 'A value of -2 indicates the special case of the default SSL  values for the initialization, Both "number_annotations_per_image" and "percentage_labeled_data" have to be -1 in this case.'
            assert FLAGS.number_annotations_per_image == -2, 'A value of -2 indicates the special case of the default SSL values for the initialization, Both "number_annotations_per_image" and "percentage_labeled_data" have to be -1 in this case.'

            #  percentage, number
            self.init_values = [
                (0.1,1),(0.2,1),(0.5,1),(1,1), # semi supervised
                # (1,3),(1,5),(1,10), # multiple
                # (0.1,10),(0.2,5),(0.5,2) # semi-supervised + multiple
            ]
        elif FLAGS.percentage_labeled_data == -3 or FLAGS.number_annotations_per_image == -3:
            #  percentage, number
            self.init_values = [
                (0.1,10),(0.2,5),(0.5,2) # semi-supervised + multiple
            ]
        elif FLAGS.percentage_labeled_data == -4 or FLAGS.number_annotations_per_image == -4:
            #  percentage, number
            self.init_values = [
                (1,10),(1,5),(1,3), (1,1), # multiple
                (0.99, 10), (0.99, 5), (0.99, 3), (0.99, 1),  # fake repetition
                (0.98, 10), (0.98, 5), (0.98, 3), (0.98, 1)  # fake repetition
            ]
        else:
            self.init_values = [
                (FLAGS.percentage_labeled_data, FLAGS.number_annotations_per_image)
            ]


        if v_fold == -1:
            self.slices = [1, 2, 3] # 4, 5
        else:
            assert 1 <= v_fold <= 5
            self.slices = [v_fold]


        # normally hard labels are used
        self.mode = 'hard'

        # setup logging
        self.report = DCICReport()



        # this setting is ONLY allowed for evaluation and used to generate additional information
        # DO NOT USE THIS OPTION
        self.cheating = False

    def get_initial_annotations(self, old_ds : DatasetDCICJson, new_ds : DatasetDCICJson, oracle: AnnotationOracle, dataset_infos : DatasetSkeleton, v_fold : int, num_annos_per_image: int, percentage_labeled: float):
        """
        get the initial annotations before the algorithm starts
        :return:
        """

        print(f"# Get initial annotations for {dataset_infos.name} and validation fold {v_fold}")

        # setup random
        r = Random()
        r.seed(v_fold)

        # check all assertation
        assert oracle.get_classes() == dataset_infos.classes, "Mismatch between classes from oracle and dataset"
        assert oracle.get_classes() == old_ds.classes, "Mismatch between classes from oracle and dataset"
        assert oracle.get_classes() == new_ds.classes, "Mismatch between classes from oracle and dataset"
        assert oracle.datasetDCIC == new_ds

        # calculate the number of samples
        files_possibly_supervised = np.array([path for path, split, _, _ in old_ds.get_image_iterator() if split in ['train','val']])
        supervised_ids = np.arange(len(files_possibly_supervised))
        r.shuffle(supervised_ids)
        supervised_ids = supervised_ids[:int(len(files_possibly_supervised) * percentage_labeled)]
        files_supervised = files_possibly_supervised[supervised_ids]


        # iterate over old_ds and create new_ds
        for path, split, _, _ in old_ds.get_image_iterator():
            org_split = split
            if split in ['train', 'val']:
                # determine if supervised
                if path in files_supervised:
                    # use the oracle to determine an estimate of the gt label
                    soft_gt = oracle.get_soft_gt(path, num_annos_per_image)
                else:
                    # unlabeled
                    split = 'unlabeled'
                    soft_gt = []

            elif split == 'test':
                # use the complete knowledge of the oracle as gt label
                # oracle.get_soft_gt(path, -1)
                # not allowed to use any annotations
                soft_gt = []

                if self.cheating:
                    # add information on test data for input metric calculcation on the test set
                    prob = oracle.get_soft_gt(path,-1)
                    soft_gt = np.zeros((len(prob)))
                    for i in range(num_annos_per_image):
                        label = oracle.r.choices(range(len(prob)), weights=prob, k=1)
                        soft_gt[label] += 1
                    soft_gt /= np.sum(soft_gt)

            else:
                raise ValueError(f"The specified split {split} is invalid")

            # hard_gt = np.argmax(soft_gt)
            # hard_label = cl[hard_gt]

            new_ds.add_image(path, split, list(soft_gt), info={'original_split': org_split})

        return new_ds



    def apply_algorithm(self):
        """
        apply the algorithm to the given data (all datasets and slices), 1. get the initial annotation 2. run the main code
        :return:
        """


        for folder_name in self.folder_dataset_names:
            # the folder name is automatically the dataset name

            assert folder_name in get_all_dataset_infos(), f"Folder name {folder_name} must be a dataset name"
            dataset_info = get_all_dataset_infos()[folder_name]


            for percentage_labeled, num_annos in self.init_values:



                oracle = AnnotationOracle(join(dataset_info.raw_data_directory, 'annotations.json'))

                success = self.before_all_slices_processed(dataset_info)
                if not success:
                    continue

                for v_fold in self.slices:

                    file_name = f'{folder_name}-slice{v_fold}.json'

                    print(f"# Start processing {file_name}")


                    # setup dataset
                    raw_ds = DatasetDCICJson.from_file(join(dataset_info.input_directory, file_name))
                    dataset_name = raw_ds.dataset_name
                    assert folder_name == dataset_name, f"Mismatch between folder name {folder_name} and dataset name {dataset_name}"
                    assert dataset_info.name == dataset_name, f"Mismatch between loaded dataset info name {dataset_info.name} and dataset name {dataset_name}"
                    v_fold = raw_ds.v_fold


                    # setup oracle and relabeled dataets
                    new_ds = DatasetDCICJson.from_definition(dataset_name, v_fold, dataset_info.classes)
                    oracle.init(new_ds)

                    # ensure that all underling distributions are of the same classes
                    assert oracle.get_classes() == raw_ds.classes
                    assert oracle.get_classes() == new_ds.classes
                    assert oracle.get_classes() == dataset_info.classes

                    # Get initial annotations

                    print(f"# Get initial annotations for  {file_name}")
                    new_ds = self.get_initial_annotations(raw_ds, new_ds, oracle,dataset_info,v_fold, num_annos, percentage_labeled)

                    # Apply main algorithm
                    print(f"Apply {self.name} to {file_name} with {num_annos} annotations on {percentage_labeled} of the data")
                    try:
                        new_ds = self.run(new_ds,oracle, dataset_info,v_fold,num_annos,percentage_labeled)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        new_ds = None

                    if new_ds is None:
                        continue


                    self.after_run_finished(new_ds)


                    b = new_ds.budget
                    new_folder =  f"{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}"
                    new_file = f'{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}.json'

                    new_ds.save_jsons(join(dataset_info.root_directory, FLAGS.output_folder, new_folder, new_file))

                self.after_all_slices_processed(dataset_info.name, percentage_labeled, num_annos)



    def run(self, ds : DatasetDCICJson, oracle: AnnotationOracle, dataset_info: DatasetSkeleton, v_fold: int, num_annos,percentage_labeled):
        """
        run the main loop to update the labels in the new_ds,
        :param ds: dataset with initialized annotations
        :param oracle: simulate annotations from users
        :return:
        """

        raise NotImplementedError("No Run Method Implmented")

    def before_all_slices_processed(self, folder : str):
        """
        callback before processing all slices from a dataset
        :param folder: Should be the dataset_name
        :return: true if sucessful
        """
        return True

    def after_all_slices_processed(self, dataset_name : str, percentage_labeled, num_annos):
        """
        callback after processing all slices from a dataset
        :param folder: Should be the dataset_name
        :return:
        """

        report_name = f"{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}"

        self.report.summarize_and_reset(report_name, self.mode, verbose=0)

    def after_run_finished(self, ds : DatasetDCICJson):
        # Show evaluations
        self.report.end_run(ds, None, None, verbose=0)