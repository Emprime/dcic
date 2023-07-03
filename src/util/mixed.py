import os
from os.path import join


from absl import flags

from src.util.const import get_all_dataset_infos

FLAGS = flags.FLAGS

# define flags
flags.DEFINE_integer(name='input_size', default=96, help='default input size of the images if it can not be determined automatically')
# flags.DEFINE_integer(name='number_classes', default=10, help='default number of classes if it can not be determined automatically')


def GetDatasetNames(dataset_name_flag):
    """
    Get the dataset name from the given flag, all is replaced with a list of datasets
    :param dataset_name_flag:
    :return:
    """
    if len(dataset_name_flag) == 1 and dataset_name_flag[0] == 'all':
        datasets = sorted(list(get_all_dataset_infos().keys()))
    else:
        datasets = dataset_name_flag

    return datasets

def get_all_dataset_files(root, folder):
    """
    Get all dataset files (splits) in the given root under the given folder name
    :param root:
    :param folder:
    :return:
    """
    # get input dataset files
    input_datasets = sorted([f for f in os.listdir(join(root, folder)) if
                             os.path.isfile(join(root, folder, f))])

    print("Found the dataset files: %s in the folder %s " % (input_datasets, join(root, folder)))

    return input_datasets

