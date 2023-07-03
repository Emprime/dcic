import itertools
import json
import time
from typing import Tuple, Iterator, List

import shutil
from os.path import join
from random import Random

from absl import app
from absl import flags
import os
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import datetime
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(name='root_directory_predictions', required=True,
                     help='The path for the root directory of all predictions which should be processed. It is expected that the predictions for all slices are stored in one directory in this root directory.', default=None)
flags.DEFINE_multi_string(name='subset', required=True,
                     help='The predictions are stored into subsets. Use the given list of subset names to determine the subsets which should be used.', default=None)
flags.DEFINE_bool(name='enable_only_overclustering_confidences', help="uses only confidences for the overclustering head and not the predefined ones, used for special spa based on confiences", default=False)



class PredictionsImport(AlgorithmSkelton):
    """
    Class to import manual predictions from other repositories for the evaluation
    """

    def __init__(self, name, files,subset_names, report=None):

        # add first subset_name to name
        name += f"_{subset_names[0]}"

        if FLAGS.enable_only_overclustering_confidences:
            name += "_only_over"


        AlgorithmSkelton.__init__(self, f'{name}')
        self.files = files
        self.subset_names = subset_names
        if report is not None:
            self.report = report

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):

        num_classes = dataset_info.num_classes

        # iterate over old split

        # try to find a corresponding file
        # files should have structure ...#dataset-<DATASET_NAME>- ... -<NUM_ANNOS>-<PERCENTAGE_LABELED>-<SPLIT_NUMBER>.1@-1-10.json
        # e.g. predictions-606-mean-no-large#dataset-Benthic-no_alg-01-0.10-3.1@-1-10.json
        file = None

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

        predictions = PredictionJson.from_file(join(FLAGS.root_directory_predictions,file))

        # get all soft gt and predictions for fuzzy anaylsis
        local_gt, train_gt = [] ,[]
        local_pred, train_pred = [],[]
        local_paths, train_paths = [],[]

        # iterate over all subset names which should be used
        for subset_name in self.subset_names:
            if subset_name in predictions.jsons:

                # for path, preds, soft_gt in zip(local_paths, local_pred, local_gt):
                for path, _, class_label, confidences in predictions.get_iterator_for_set(subset_name):
                    # # ensure class label and confidences are of the same type
                    # print(class_label, confidences, dataset_info.classes)

                    # ensure either cluster label (more confidences then classes) or class label and confindences match the known class structure
                    assert len(confidences) != len(dataset_info.classes) or dataset_info.classes[np.argmax(confidences, axis=-1)] == class_label, "Class label and given classes do not match"

                    soft_gt = oracle.get_soft_gt(path, -1)

                    # determine split and gt based on original splits
                    split = ds.get(path, 'original_split')

                    # print(confidences)
                    confidences = np.array(confidences)


                    # determine mapping between clusters and classes if applicable
                    if "fuzzy" in subset_name:
                        local_gt.append(soft_gt)
                        local_pred.append(confidences)
                        local_paths.append(path)

                        # check for training
                        if "train" in subset_name or "val" in subset_name:
                            train_gt.append(soft_gt)
                            train_pred.append(confidences)
                            train_paths.append(path)
                    else:
                        if FLAGS.enable_only_overclustering_confidences:
                            estimated_distribution = np.zeros(len(soft_gt))
                            estimated_distribution[np.argmax(soft_gt)] = 1
                        else:
                            estimated_distribution = confidences
                        ds.update_image(path, split, list(estimated_distribution))


        # convert fuzzy images
        if len(local_pred) > 0:


            # found fuzzy images
            local_gt = np.array(local_gt)
            train_gt = np.array(train_gt)
            local_pred = np.array(local_pred)
            train_pred = np.array(train_pred)
            local_paths = np.array(local_paths)

            # use only training for mapping
            mapped_pred, mapping = Combine_clusters_by_purity(np.argmax(train_gt, axis=-1),
                                                              np.argmax(train_pred, axis=-1), return_mapping=True)

            mapping = dict(mapping)

            print("Fuzzy clusters mapping", mapping, )  # , mapped_pred
            # local_pred = mapped_pred

            # detect pred clusters without mapping
            pred_clusters = np.unique(np.argmax(local_pred, axis=-1))
            not_found_clusters = [c for c in pred_clusters if c not in mapping]
            print("Not found clusters in training: ", not_found_clusters)
            for c in not_found_clusters:
                mapping[c] = oracle.r.choice([i for i in range(num_classes)])


            # iterate over clusters
            for cluster, assigned_class in mapping.items():
                where = np.where(np.argmax(train_pred, axis=-1) == cluster)

                # print(where, np.argmax(local_pred, axis=-1))
                # print(local_gt.shape)

                # calculate purity of clusters
                gt_in_cluster = train_gt[where]
                # print(gt_in_cluster)

                if len(gt_in_cluster) > 0:

                    found_classes, counts = np.unique(np.argmax(gt_in_cluster, axis=-1), return_counts=True)
                    total = np.sum(counts)
                    percentage_of_cluster = dict(zip(found_classes,[c/total for c in counts]))

                else:
                    percentage_of_cluster = {assigned_class:1}

                # print(cluster, percentage_of_cluster)

                where = np.where(np.argmax(local_pred, axis=-1) == cluster)
                for path, soft_gt in zip(local_paths[where], local_gt[where]):


            # for path, proposed_class, soft_gt in zip(local_paths, local_pred, local_gt):


                    split = ds.get(path, 'original_split')

                    # make one hot encoded # TODO not realistic
                    # temp = np.zeros(len(soft_gt))
                    # temp[proposed_class] = 1
                    # estimated_distribution = temp
                    estimated_distribution = [percentage_of_cluster.get(i,0) for i in range(num_classes)]

                    ds.update_image(path, split, list(estimated_distribution))

        # num_items_not_test = float(len([1 for _, split, _, _ in ds.get_image_iterator() if split != 'test']))
        # print(f"Final Budget: {ds.weighted_budget/num_items_not_test} / {ds.budget/num_items_not_test} [{num_items_not_test}]")
        # print(f"Final KL:")
        # for split in ['train','val','test','unlabeled']:
        #     paths, estimated_gts = ds.get_training_subsets(split)
        #     real_gt = []
        #     for path,estimated_gt in zip(paths,estimated_gts):
        #         soft_gt = oracle.get_soft_gt(path, -1)
        #         real_gt.append(soft_gt)
        #
        #     real_gt, estimated_gts = np.array(real_gt), np.array(estimated_gts)
        #     # print(real_gt.shape,estimated_gts.shape)
        #     if len(real_gt) > 0  and len(estimated_gts[0]) > 0:
        #         print(f"{split} - {self.kl_div(real_gt,estimated_gts)}")
        #     else:
        #         print(split, f"- found {len(real_gt)} Entries")

        return ds


    # def kl_div(self,y_true, y_pred, epsilon = 0.00001):
    #     """
    #        own kullback  loss implementation
    #        :param y_true:
    #        :param y_pred:
    #        :return:
    #     """
    #     # seems to be identical to keras loss
    #     y = np.clip(y_true, epsilon, 1 - epsilon)
    #     x = np.clip(y_pred, epsilon, 1 - epsilon)
    #
    #     kl = np.sum(y * np.log(y / x), axis=1)
    #
    #     return np.mean(kl)


def Combine_clusters_by_purity(y_true,y_pred, return_mapping = False, mapping=None):
    """
    combine more clusters to number of clusters of gt, use only first n examples for gt propagation
    :param y_true:
    :param y_pred:
    :return:
    """
    assert not (y_true == None).any() and not (y_pred == None).any()

    # print(y_true.shape, y_pred.shape)

    # filter for both valid values
    # y_true_masked, y_pred_masked, mask = mask_Nones(y_true, y_pred, return_mask=True)

    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)

    # used for indexing if y-true and y-pred are not containing values like [0,1,2,3,..]
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)

    if mapping is None:

        raw_mapping = np.argmax(contingency_matrix, axis=0)

        # calculate mapping
        mapping = []
        for x, y in enumerate(raw_mapping):
            mapping.append((clusters[x],classes[y]))

    result = np.zeros(y_pred.shape)
    unmapped_clusters = list(np.copy(clusters))
    for x,y in mapping:
        np.putmask(result,y_pred == x,y)
        if x in unmapped_clusters:
            unmapped_clusters.remove(x)

    for unmapped_cluster in unmapped_clusters:
        np.putmask(result, y_pred == unmapped_cluster, -1)

    result = result.astype(int)

    if return_mapping:
        return result, mapping
    else:
        return result

def mask_Nones(dist_x, dist_y, file_names=None, return_mask=False, return_number=False):
    """
    mask the elements which are none in dist x or dist y
    :param dist_x:
    :param dist_y:
    :param file_names: return the file names corresponding to the dists, will return as last value the file names
    :param return_mask:
    :param return_number:
    :return:
    """
    if return_number:
        number = -1
        dist_x = dist_x.copy()
        dist_y = dist_y.copy()
        dist_x[dist_x == None] = number
        dist_y[dist_y == None] = number

        # print((dist_x))
        # print((dist_y))

        if file_names is None:
            return dist_x.astype(int), dist_y.astype(int)
        else:
            return dist_x.astype(int), dist_y.astype(int), file_names.copy()

    mask = (dist_x != None) & (dist_y != None)
    dist_x = dist_x[mask].astype(int)
    dist_y = dist_y[mask].astype(int)

    if file_names is not None:
        # print(mask)
        # print(file_names)
        file_names = np.array(file_names)[mask]

        if return_mask:
            return dist_x, dist_y, mask, file_names
        else:
            return dist_x, dist_y, file_names
    else:

        if return_mask:
            return dist_x, dist_y, mask
        else:
            return dist_x, dist_y


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

    def get_prediction_iterator(self) -> Iterator[Tuple[str, str, str, datetime.datetime, str, int, str, List[float]]]:
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

    def get_iterator_for_set(self, sub_name) -> Iterator[Tuple[str, int, str, List[float]]]:
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

    used_overclustering = any(["fuzzy" in s or "combined" in s for s in FLAGS.subset])

    # use all if no simulation
    # or use s2c2 with overclustering subsets or not with normal subsets

    # uses actually the normal head for the overclustering
    enable_special_overclustering = False


    report = None

    if not enable_special_overclustering and not used_overclustering:
        print("Import Meanteacher no proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "mean-no" in file])
        alg = PredictionsImport('mean', files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if enable_special_overclustering or used_overclustering:
        print("Import Meanteacher with proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "mean-s2c2" in file])
        alg = PredictionsImport('mean_s2c2' + ("_special" if enable_special_overclustering else ""), files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if not enable_special_overclustering and not used_overclustering:
        print("Import PI no proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "pi-no" in file])
        alg = PredictionsImport('pi', files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if enable_special_overclustering or used_overclustering:
        print("Import PI with proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "pi-s2c2" in file])
        alg = PredictionsImport('pi_s2c2'+ ("_special" if enable_special_overclustering else ""), files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if not enable_special_overclustering and not used_overclustering:
        print("Import PseudLabel no proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "pseudo-no" in file])
        alg = PredictionsImport('pseudo_ssl', files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if enable_special_overclustering or used_overclustering:
        print("Import PseudLabel with proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "pseudo-s2c2" in file])
        alg = PredictionsImport('pseudo_s2c2'+ ("_special" if enable_special_overclustering else ""), files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if not enable_special_overclustering and not used_overclustering:
        print("Import FixMatch no proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "fixmatch-no" in file])
        alg = PredictionsImport('fixmatch', files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report

    if enable_special_overclustering or used_overclustering:
        print("Import FixMatch with proposals")
        files = sorted([file for file in os.listdir(root_directory_predictions) if "fixmatch-s2c2" in file])
        alg = PredictionsImport('fixmatch_s2c2'+ ("_special" if enable_special_overclustering else ""), files, subset_names=FLAGS.subset, report=report)
        alg.apply_algorithm()
        report = alg.report


    if report is not None:
        report.show()


if __name__ == '__main__':
    app.run(main)