import glob

import shutil
from os.path import join

from absl import flags
import os
from absl import app
from tqdm import tqdm

from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson
from src.util.oracle import AnnotationOracle

FLAGS = flags.FLAGS
flags.DEFINE_string(name='file',
                    help='full path to the dataset json which should be visualized', required=True, default=None)

flags.DEFINE_string(name='visualize_path',
                    help='full path to the output directory', default="/data/visualize")

flags.DEFINE_boolean(name='show_soft_label',
                     help='extends the file name with the softlabel for a better visualization. This also means that these files should not for export because an import requires the same unique filesnames', default=False)


def main(argv):
    """
    Visualize a given dataset json file
    :param argv:
    :return:
    """
    file_path = FLAGS.file

    if "*" in file_path:
        file_paths = sorted(glob.glob(file_path))
    else:
        file_paths = [file_path]

    for file_path in file_paths:
        file_name = file_path.split("/")[-1].split(".json")[0]
        visualize_path = join(FLAGS.visualize_path, file_name)
        if os.path.exists(visualize_path):
            shutil.rmtree(visualize_path)
        os.makedirs(visualize_path)
        print("Visualize the file %s in the folder %s" % (file_path, visualize_path))

        ds = DatasetDCICJson.from_file(file_path)
        dataset_info = get_all_dataset_infos()[ds.dataset_name]

        for path, split, hard_label, soft_label in tqdm(ds.get_image_iterator()):
            split = str(split)
            hard_label = dataset_info.classes[hard_label]
            os.makedirs(join(visualize_path, split, hard_label), exist_ok=True)
            if FLAGS.show_soft_label:
                new_file_name = "%s-%s.png" % (path.split("/")[-1].split(".png")[0],
                                                                            ["%0.02f" % l for l in soft_label])
            else:
                new_file_name = path.split("/")[-1]


            shutil.copy(join(dataset_info.raw_data_root_directory, path), join(visualize_path, split, hard_label,
                                                             new_file_name))


if __name__ == '__main__':
    app.run(main)