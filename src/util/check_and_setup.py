from absl import app
from src.util.const import get_all_dataset_infos
import os

def main(args):

    for dataset_name, dataset_class in get_all_dataset_infos().items():
        print(f"Check & setup dataset {dataset_name}")
        # check naming is correct
        assert dataset_name == dataset_class.name
        path = dataset_class.root_directory + 'raw_datasets/' + dataset_name
        if os.path.exists(path):
            # check and setup
            dataset_class.check_setup()
        else:
            print(f"Check & setup dataset failed for {dataset_name}. The folder {path} does not exist.")


if __name__ == '__main__':
    app.run(main)
