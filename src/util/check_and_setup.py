from absl import app
from src.util.const import get_all_dataset_infos, get_lite_dataset_infos
import os

def main(args):

    for dataset_name, dataset_class in sorted(get_lite_dataset_infos().items()):
        print(f"Check & setup dataset {dataset_name}")
        # check naming is correct
        assert dataset_name == dataset_class.name
        path = dataset_class.root_directory + 'raw_datasets/' + dataset_name

        # check and setup
        dataset_class.check_setup()


if __name__ == '__main__':
    app.run(main)
