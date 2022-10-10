import json
import time
from typing import Tuple, Iterator

import shutil
from os.path import join
from random import Random

from absl import app
from absl import flags
import os
import numpy as np
from tqdm import tqdm
import datetime
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from src.util.const import get_all_dataset_infos
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(name='root_directory_predictions', required=True,
                     help='The path for the root directory of all predictions which should be processed. It is expected that the predictions for all slices are stored in one directory in this root directory.', default=None)
flags.DEFINE_multi_string(name='subset', required=True,
                     help='The predictions are stored into subsets. Use the given list of subset names to determine the subsets which should be used.', default=None)

class PredictionsImport(AlgorithmSkelton):
    """
    Class to import manual predictions from other repositories for the evaluation
    """

    def __init__(self, name, files,subset_names):
        AlgorithmSkelton.__init__(self, f'{name}')

        self.files = files
        self.subset_names = subset_names

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):
        # iterate over old split

        # try to find a corresponding file
        # files should have structure ...#dataset-<DATASET_NAME>- ... -<NUM_ANNOS>-<PERCENTAGE_LABELED>-<SPLIT_NUMBER>.1@-1-10.json
        # e.g. predictions-606-mean-no-large#dataset-Benthic-no_alg-01-0.10-3.1@-1-10.json
        file = None

        # print(self.files)
        for _file in self.files:
            temp = _file.split("#dataset-")[-1].split(".1@-1-10.json")[0]
            temp = temp.split("-")
            # print(temp)
            dataset_name, _, _num_annos, _percentage_labeled, _v_fold = temp
            if dataset_name == dataset_info.name and \
                    int(_num_annos) == num_annos and\
                    abs(float(_percentage_labeled) - percentage_labeled) < 0.01 and \
                    int(_v_fold) == v_fold:
                file = _file
                break


        if file is None:
            print(f"WARNING: Import for {dataset_info.name} and {num_annos} annotations for {percentage_labeled} amount of the data with v_fold {v_fold} NOT possible!!!")
            return None


        # print(f"Convert predictions from file {file} for {dataset_info.name} {v_fold}")


        predictions = PredictionJson.from_file(join(FLAGS.root_directory_predictions,file))

        # iterate over all subset names which should be used
        for subset_name in self.subset_names:
            if subset_name in predictions.jsons:
                for path, _, _, confidences in predictions.get_iterator_for_set(subset_name):

                    # determine split and gt based on original splits
                    split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
                    # expects the same unique file name in path like in the original
                    ds.update_image(path, split, confidences)

        return ds



class PredictionJson:
    def __init__(self, config, ID_experiment, jsons ):
        # self.dataset = config.dataset

        if config is not None:
            # print(config, type(config))
            if not isinstance(config,str):
                config = dict(vars(config))
                config_json = json.dumps({i: config[i] for i in config if i != 'graph'})
            else:
                config_json = config
        else:
            config_json = "Not available"
        self.config_json = config_json

        self.jsons = jsons
        self.ID_experiment = ID_experiment # identifies the experiment the results belong to, should be unqiue


    def save_json_list(self, out_path):
        with open(join(out_path, 'predictions-%s.json' % (self.ID_experiment)), 'w') as outfile:
            json.dump(list(self.jsons.values()), outfile)

    def get_prediction_iterator(self) -> Iterator[Tuple[str, str, str, datetime.datetime, str, int, str, float]]:
        """
        return iterator over predictions with config as json, identifier, name,  created_at, image_path, cluster_label, class_label, confidence
        :return:
        """

        for json in self.jsons.values():
            for pred in json['predictions']:
                yield json['config'], json['identifier'], json['name'], json['created_at'],  pred['image_path'], pred['cluster_label'], pred['class_label'], pred.get('confidence',-1)

    def get_general_info_for_set(self,sub_name):
        json = self.jsons[sub_name]

        return  json['config'], json['identifier'], json['name'], json['created_at']
    @property
    def length(self):
        return len(self.jsons)

    def get_iterator_for_set(self, sub_name) -> Iterator[Tuple[str, int, str, float]]:
        """
        Get iterator based on the index of the internal sets, image_path, cluster_label, class_label, confidence
        :param index:
        :return:
        """

        json = self.jsons[sub_name]
        for pred in json['predictions']:
            yield  pred['image_path'], pred['cluster_label'], pred['class_label'], pred.get('confidences',-1)


    @classmethod
    def from_definition(cls, config, ID_experiment):
        return PredictionJson(config,ID_experiment, {})

    @classmethod
    def from_file(cls, prediction_file):
        with open(prediction_file, 'r') as file:
            jsons = json.load(file)

            assert len(jsons) > 0, "empty file found"

            json_dict = {j['name'] : j for j in jsons}

            return PredictionJson(jsons[0]['config'], cls.get_identifier_from_file(prediction_file), json_dict)

    @classmethod
    def get_identifier_from_file(cls, file_name):
        """
        get identifier from raw file name, take last element from directory seperators and cut predicitons and .json
        :param file_name:
        :return:
        """
        return file_name.split("/")[-1].split("predictions-")[-1].split(".json")[0]


def main(argv):
    """
       COnvert the predictions of an external tool into useable datasets for the evaluation
       :return:
    """

    root_directory_predictions = FLAGS.root_directory_predictions

    # define which files belong to which imported method
    # files should have structure ...#dataset-<DATASET_NAME>- ... -<NUM_ANNOS>-<PERCENTAGE_LABELED>-<SPLIT_NUMBER>.1@-1-10.json
    # e.g. predictions-606-mean-no-large#dataset-Benthic-no_alg-01-0.10-3.1@-1-10.json

    files = sorted([file for file in os.listdir(root_directory_predictions) if "mean-no" in file])
    alg = PredictionsImport('mean', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "mean-s2c2" in file])
    alg = PredictionsImport('mean_s2c2', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "pi-no" in file])
    alg = PredictionsImport('pi', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "pi-s2c2" in file])
    alg = PredictionsImport('pi_s2c2', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "pseudo-no" in file])
    alg = PredictionsImport('pseudo_ssl', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "pseudo-s2c2" in file])
    alg = PredictionsImport('pseudo_s2c2', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "fixmatch-no" in file])
    alg = PredictionsImport('fixmatch', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()

    files = sorted([file for file in os.listdir(root_directory_predictions) if "fixmatch-s2c2" in file])
    alg = PredictionsImport('fixmatch_s2c2', files, subset_names=FLAGS.subset)
    alg.apply_algorithm()









if __name__ == '__main__':
    app.run(main)