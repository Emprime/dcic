import glob
import hashlib
import os
import random
from random import Random
import matplotlib.pyplot as plt
import cv2
import datetime
import json
from os.path import join

from pandas import DataFrame
from typing import List, Iterator, Tuple, Optional

import numpy as np
from tqdm import tqdm
import pandas as pd

import time

from src.util.json import DatasetDCICJson



class AnnotationOracle:

    def __init__(self, annotation_file : str,):
        """
        Create oracle
        :param annotation_file: file location of the used annotations
        :param datasetDCIC: dataset the oracle should be used for, ensure that budget is not lost
        """
        self.annoJson = AnnotationJson.from_file(annotation_file)
        self.paths, self.classes, self.prob_data = self.annoJson.get_probability_data()
        # self.budget = 0
        # self.weighted_budget = 0

        # save for bookkeeping on saving in dataset
        self.datasetDCIC = None


        self.r = Random()
        self.r.seed(49312)

    def init(self,  datasetDCIC : DatasetDCICJson):
        """
        Call this init before any requests to the oracle, resets also the internal seeding
        :param datasetDCIC: the dataset which the oracle should be used for
        :param newdatasetDCIC: optional dataset the budget should be transferred to
        :return:
        """

        self.r.seed(49312)

        # self.budget = datasetDCIC.budget
        # self.weighted_budget = datasetDCIC.weighted_budget

        # save for bookkeeping on saving in dataset
        self.datasetDCIC = datasetDCIC

        # convert all paths to unique id
        paths = []
        for p in self.paths:
            unique_path = self.datasetDCIC.extract_unique_id_from_path(p)
            if unique_path in paths:
                print(self.paths)
            assert unique_path not in paths, f"Duplicates are not allowed for the unique id {unique_path} in {paths}"
            paths.append(unique_path)



        self.paths = paths


    def get_classes(self):
        """
        get the classes of the annotations
        :return:
        """

        return self.classes

    def get_soft_gt(self, path : str, number_annotations_per_image: int):
        """
        get a soft gt vector approximated based on the number of annotations, negative values will return the complete knowledge must only be used for evaluation
        :param number_annotations_per_image:
        :return:
        """

        assert self.datasetDCIC is not None , "Oracle is not initialized, call init(dataset)"

        path = self.datasetDCIC.extract_unique_id_from_path(path)

        if number_annotations_per_image < 0:
            # get complete groundtruth,
            return self.prob_data[self.paths.index(path)]
        else:
            # combined_anno = None
            # for i in range(number_annotations_per_image):
            combined_anno = self.get_annotation(path, num_anno=number_annotations_per_image)
            return combined_anno / np.sum(combined_anno)



    def get_annotation(self, path, weight=1, num_anno=1):
        """
        simulate selecting an annotation from the underyling distribution,
        you can define a different weight to calculate the budget
        :param path:
        :return:
        """

        assert self.datasetDCIC is not None, "Oracle is not initialized, call init(dataset)"

        path = self.datasetDCIC.extract_unique_id_from_path(path)

        prob = self.prob_data[self.paths.index(path)]
        anno = np.zeros((len(prob)))
        labels = self.r.choices(range(len(prob)),weights=prob,k=num_anno)
        for l in labels:
            anno[l] += 1

        # print(file,prob,anno)

        self.datasetDCIC.budget += num_anno
        self.datasetDCIC.weighted_budget += num_anno * weight

        return anno





class AnnotationJson:
    def __init__(self, dataset_name, anno_table : Optional[DataFrame], anno_json):
        """

        The Annotation Aggregation can either be init with an table of annotations, pd_dataframe, filename x classes, and a user who created the annotations
        or by a list of Annotation lists

        :param dataset_name: name of the dataset
        :param anno_table:
        :param anno_json:
        """

        self.dataset_name = dataset_name

        self.anno_table : DataFrame = anno_table
        self.anno_json = anno_json

    @classmethod
    def from_pandas_table(cls, dataset_name, anno_table, user, id_to_filename_format="%s", set_name="generated"):
        """

        :param anno_table: filename x classes,
        :param user:
        :param id_to_filename_format: format string to convert the id to a file_name
        :return:
        """

        # case 2: annotation table is provided
        assert anno_table is not None
        assert user is not None
        # -> create an json list

        print("convert annotation table to list")

        ids = list(anno_table.index.values)
        labels_for_dataset = list(anno_table.columns)

        # cast table of annotations to a list of annotations per image, mind that not all images have the same number of annotations
        all_annotations = []
        now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        for id in tqdm(ids):

            annotations_per_object = 0
            for label in labels_for_dataset:
                number_annotations = int(anno_table.loc[id, label])

                for i in range(number_annotations):
                    # index = i + annotations_per_object
                    # annotations = all_annotations[index] if index in all_annotations else []
                    all_annotations.append(
                        {"image_path": id_to_filename_format % id, "class_label": label,
                         "created_at": now})

                    # all_annotations[index] = annotations

                annotations_per_object += number_annotations

        json_annotations = []
        # for key in all_annotations:
        name = "%s-%s" % (dataset_name, set_name)

        json_anno = {"name": name, "user_mail": user, "annotation_time": 0.0, "dataset_name": dataset_name,
                     "annotations": all_annotations}
        json_annotations.append(json_anno)

        return AnnotationJson(dataset_name, anno_table, json_annotations)


    @classmethod
    def from_file(cls, annotation_file, check_for_annominous=True):
        """
        Load annotation file and return handler
        :param annotation_file:
        :return:
        """

        with open(annotation_file, 'r') as outfile:

            annotation_jsons = json.load(outfile)

            # case 1: annotation json list

            assert len(annotation_jsons) > 0 , "File  must contain at least one annotations set to loadable otherwise dataset can't be determined"

            dataset_name = annotation_jsons[0]['dataset_name']

            # -> create table
            print("convert annotation jsons to table")
            # get names and labels
            img_names = []
            labels = []

            for annos in annotation_jsons:

                # safe guard for anonimous data
                if check_for_annominous:
                    assert annos.get('user_mail','INVALID') == 'INVALID'
                    assert "INVALID" in annos.get('name','INVALID')

                for entry in annos["annotations"]:

                    # add only valid annotations to table
                    if entry["class_label"] is not None:

                        # if entry["image_path"] not in img_names:
                        img_names.append(entry["image_path"])

                        # if entry["class_label"] not in img_names:
                        labels.append(entry["class_label"])

            img_names = list(np.unique(np.array(img_names)))
            labels = list(np.unique(np.array(labels)))

            # fast access maps
            map_names = dict(zip(img_names, list(np.arange(0, len(img_names)))))
            map_labels = dict(zip(labels, list(np.arange(0, len(labels)))))

            data = np.zeros((len(img_names), len(labels)))

            for annos in tqdm(annotation_jsons):
                for entry in annos["annotations"]:
                    # add only valid annotations to table
                    if entry["class_label"] is not None:
                        data[map_names[entry["image_path"]], map_labels[entry["class_label"]]] += 1


            return AnnotationJson(dataset_name, pd.DataFrame(data=data, index=img_names, columns=labels), annotation_jsons)



    @classmethod
    def empty(cls, dataset_name):
        """
        Be aware that no pandas table is used
        :param dataset_name:
        :return:
        """
        return AnnotationJson(dataset_name, None, [])


    def map_to_new_directory(self, dataset_name, original_directory,  new_directory):
        """
        Will discard any images which are not in the directory and the annotation
        :param original_directory: give a directory which was used to create these annotations
        :param new_directory: give a directory to which the annotations should be mapped, must end with / and only .png images are used
        :return:
        """

        print("map to new directory ...")

        # save and reset internals
        original_anno_json = self.anno_json.copy()
        self.anno_json = []
        self.anno_table = None
        self.dataset_name = dataset_name

        assert new_directory.endswith(dataset_name + "/"), "By convention the new directory should be folder of dataset" \
                                                           " and the name should be the dataset_name " \
                                                           "and directory should end with /," \
                                                           " but %s and %s was given" % (dataset_name, new_directory)

        path_map = {}
        for anno_set in original_anno_json:
            print("parse set %s " % anno_set['name'])

            # calculate hash values for original annotations
            # dict with hashes as key and list of images as value
            print("generated hashes ...")
            hash_map = {}
            for anno_entry in tqdm(anno_set['annotations']):
                old_img_path = anno_entry["image_path"]
                load_path = join(original_directory, old_img_path)
                if load_path in path_map:
                    hash = path_map[load_path]
                else:
                    img = cv2.imread(load_path)
                    if img is not None:
                        hash = hashlib.md5(img).hexdigest()
                        path_map[load_path] = hash

                    else:
                        print("WARNING:  Image %s not found" % (load_path))
                        continue
                # put hash into hash map with values
                list_of_old_paths = hash_map.get(hash, [])
                list_of_old_paths.append((old_img_path, anno_entry['class_label'], anno_entry['created_at']))
                hash_map[hash] = list_of_old_paths

            print("generated %d hashes" % len(hash_map.keys()))

            # iterate over new images and check if entry exists
            # create table of new entries
            print("search in new directory ...")
            all_files = sorted(glob.glob(new_directory + '/**/*.png', recursive=True))
            print("found %d files" % len(all_files))
            new_annotations = []
            for file in tqdm(all_files):
                if file in path_map:
                    hash = path_map[file]
                else:
                    img = cv2.imread(file)
                    hash = hashlib.md5(img).hexdigest()
                    path_map[file] = hash

                if hash in hash_map:
                    # found imag
                    new_img_path = join(dataset_name,file.split(new_directory)[-1]) # add dataset_name
                    for old_img_path, class_label, created_at in hash_map[hash]:
                        # print("%s -> %s" % (old_img_path, new_img_path))
                        new_annotations.append((new_img_path, class_label, datetime.datetime.strptime(created_at, "%d-%m-%Y-%H-%M-%S")))

            print("reidentified %d images in new directory" % len(new_annotations))

            # set internal values to values calculated by
            # reset time because it might not be valid anymore
            self.add_annotationset(anno_set["name"], anno_set["user_mail"], 0.0 , new_annotations)

    def anoymize(self):
        for anno_set in self.anno_json:
            anno_set['name'] = "INVALID"
            anno_set['user_mail'] = "INVALID"
            anno_set['annotation_time'] = 0

    def add_annotationset(self, anno_set_name, user, annotation_time, annotations : List[Tuple[str,str, datetime.datetime]], update_summary_table=True, ignore_wrong_dataset=False, ignore_class=[]):
        """
        add a new annotation set
        :param anno_set_name:
        :param user:
        :param annotation_time:
        :param annotations: list of tuples with, imagepath, class_label and timestamp of creation (either datetime or string), class_label can be none
        :param update_summary_table: update the summary table, if you dont update, this will lead to incosistency if it is accessed later
        :return:
        """

        set_json = {
            "name": 'INVALID'+str(random.randint(0,999)), "dataset_name": self.dataset_name,
            "user_mail": 'INVALID', "annotation_time": -1, "annotations": []
        }
        for image_path, class_label, created_at in tqdm(annotations):

            assert image_path.startswith(self.dataset_name) or ignore_wrong_dataset, "By convention the image path %s should start with the dataset_name %s" % (image_path, self.dataset_name)

            if class_label is None or class_label in ignore_class:
                continue # ignore this element


            anno_json = {"image_path": image_path,
                         "created_at": created_at.strftime("%d-%m-%Y-%H-%M-%S") if not isinstance(created_at, str) else created_at,
                         "class_label": class_label}

            set_json["annotations"].append(anno_json)


            if update_summary_table:
                # add to summary table
                if self.anno_table is None:
                    self.anno_table = pd.DataFrame({class_label: []})

                if class_label not in self.anno_table.columns:
                    self.anno_table[class_label] = 0
                if image_path not in self.anno_table.index:
                    # add new row
                    # cols = list(self.anno_table.columns)
                    # ids = list(self.anno_table.index.values) + [image_path]
                    # self.anno_table = self.anno_table.append(pd.DataFrame(data=np.zeros((len(cols),1)), index=ids, columns=cols))
                    self.anno_table.loc[image_path] = 0



                # increae number
                self.anno_table.loc[image_path, class_label] += 1


        # add to list of annotations
        self.anno_json.append(set_json)


    def get_annotation_iterator(self) -> Iterator[Tuple[str, str, float, str, datetime.datetime, str]]:
        """
        Create an iterator for the annotation set
        :return: iterator of tuples anno_set_name, dataset_name, user, annotation_time, image_path, created_at, class_label
        """
        for anno_set in self.anno_json:
            for anno_entry in anno_set['annotations']:
                yield (anno_set["name"], anno_set["dataset_name"], anno_set["user_mail"], anno_set["annotation_time"], anno_entry["image_path"], anno_entry["created_at"], anno_entry["class_label"])

    def get_annotationsets(self) -> Iterator[Tuple[str, str, float, List[Tuple[str,str, datetime.datetime]]]]:
        """
        iterate over th annotation sets
        """

        for anno_set in self.anno_json:
            annos = []
            for anno_entry in anno_set['annotations']:
                annos.append((anno_entry["image_path"], anno_entry["class_label"], anno_entry["created_at"]))

            yield (anno_set["name"], anno_set["dataset_name"], anno_set["user_mail"], anno_set["annotation_time"], annos)


    def add_other_anno_file(self, annotationfile: str, ignore_wrong_dataset = False, ignore_class=[], check_for_annonimous=True):
        """
        add another annotation file to this annotation expects to have the same dataset name
        :param annotationfile:
        :param ignore_wrong_dataset: during adding the new item ignores wrong dataset warnings
        :return:
        """
        temp = AnnotationJson.from_file(annotationfile,check_for_annominous=check_for_annonimous)

        for name, dataset_name, user, time, annos in temp.get_annotationsets():
            self.add_annotationset(name,user, time, annos, ignore_wrong_dataset=ignore_wrong_dataset, ignore_class=ignore_class)

    def get_table(self):
        return self.anno_table

    def save_json_list(self, path, file_name_format = "%s.json"):
        """

        :param path:
        :param file_name_format: change the file format, needs one parameter %s to insert the dataset
        :return:
        """

        os.makedirs(path, exist_ok=True)

        with open(join(path, file_name_format % (self.dataset_name)),
                  'w') as outfile:
            json.dump(self.anno_json, outfile)

    def get_probability_data(self, return_numbers=False):
        """
        get the probabilites form the underlying dataframe, and the classes, names
        :param return_numbers: return the absolute numbers  instead of the propbability data
        :return:
        """

        if self.anno_table is None:
            return [], [], np.array([])

        # convert annotations to probabilies
        classes = list(self.anno_table.columns.values)
        imgs = list(self.anno_table.index.values)
        data = self.anno_table.to_numpy()

        if not return_numbers:
            data = data / data.sum(axis=1, keepdims=True)  # cast to probs

        return imgs, classes, data


    def print_statistics(self, out_path = None):
        """
        calcluate the annotations statistics like amount of annotations and agreement
        :return:
        """
        if out_path is not None:
            os.makedirs(out_path, exist_ok=True)

        # table of images to classes and absolute counts
        data = self.anno_table.to_numpy()
        classes = list(self.anno_table.columns.values)

        # calculate amount of annotations per image
        num_annos_per_image = data.sum(axis=1)

        counts, bins, bars = plt.hist(num_annos_per_image, bins=np.arange(-1, np.max(num_annos_per_image)+2, 1))
        print(f"Num. Annotations min {np.min(num_annos_per_image):0.02f}, max {np.max(num_annos_per_image):0.02f}, mean {np.mean(num_annos_per_image):0.02f} +- {np.std(num_annos_per_image):0.02f}")
        print("Num. Annotations per bin", dict(zip(bins, counts)))
        if out_path is not None:
            plt.savefig(join(out_path, 'num_annotations_per_image.png'))
        plt.clf()

        # calculate the class based on majority vote
        prob_data = data / data.sum(axis=1, keepdims=True)
        labels = np.argmax(prob_data,axis=1)
        cl_index, numbers = np.unique(labels, return_counts=True)
        # print(items,numbers)
        print("Images per label (majority vote)")
        for i, n in zip(cl_index, numbers):
            print("%s: %d" % (classes[i], n))
        print("Total: %d" % np.sum(numbers))

        # calculate agreement
        percentage_max = np.max(prob_data, axis=1) * 100
        counts, bins, bars = plt.hist(percentage_max, bins=np.arange(0, 101, 5), )
        print(f"Percentage agreement min {np.min(percentage_max):0.02f}, max {np.max(percentage_max):0.02f}, mean {np.mean(percentage_max):0.02f} +- {np.std(percentage_max):0.02f} [all in %]")
        print("Percentage agreement per bin ", dict(zip(bins, counts)))
        if out_path is not None:
            plt.savefig(join(out_path, 'percent_agreement.png'))
        plt.clf()


