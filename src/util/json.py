import os
import json
import numpy as np
import random
from typing import List, Iterator, Tuple


class DatasetDCICJson:

    def __init__(self, dataset_name: str, v_fold : int, classes: List[str], images: List, budget=0, weighted_budget=0):
        """
        This constructor should not be used, use the class mehtods from Definition and from File
        :param dataset_name:
        :param classes:
        :param images:
        """
        self.dataset_name = dataset_name
        self.classes = classes
        self.v_fold = v_fold
        assert v_fold in [1,2,3,4,5], "Invalid v_fold %d only the values 1 to 5 are allowed" % v_fold

        # the budget used to create this dataset
        self.budget = budget
        self.weighted_budget = weighted_budget

        # create images dictionary
        self.images_dict = {self.extract_unique_id_from_path(entry['path']) : entry for entry in images}

        assert len(self.images_dict) == len(images), "Found different number of entries %d vs. %d, " \
                                                          "check for duplicate paths" \
                                                          % (len(self.images_dict), len(images))

        # print("init with %d" % len(self.images))

    def __len__(self):
        return len(self.images_dict)

    @classmethod
    def from_definition(cls, dataset_name, v_fold, classes):
        """
        use ignore wrong dataset only when you know what you are doing, this will lead to a ẃrongly formated dataset json
        :param dataset_name: original dataset name used to create this subset
        :param classes:
        :param ignore_wrong_dataset_name:
        :return:
        """
        return DatasetDCICJson(dataset_name, v_fold, classes, [])

    @classmethod
    def from_file(cls, file_path):
        """
        load dataset json
        :param file_path:
        :return:
        """

        with open(file_path, "r") as file:
            json_content = json.load(file)

            # ensure elements are in json
            assert "name" in json_content, "Wrong json contents: %s" % json_content
            assert "v_fold" in json_content, "Wrong json contents: %s" % json_content
            assert "classes" in json_content, "Wrong json contents: %s" % json_content
            assert "images" in json_content, "Wrong json contents: %s" % json_content
            assert "budget" in json_content, "Wrong json contents: %s" % json_content
            assert "weighted_budget" in json_content, "Wrong json contents: %s" % json_content

            return DatasetDCICJson(json_content["name"], json_content["v_fold"], json_content["classes"], json_content["images"], json_content['budget'], json_content['weighted_budget'])

    def add_helper(self,  path, split,  soft_gt: List[float], info=None):
        """
        small helper function for adding and updating an image
        :param path:
        :param split:
        :param soft_gt:
        :param info:
        :return:
        """

        assert len(soft_gt) == len(self.classes) or len(soft_gt) == 0, "Expected shape of soft gt to be of length classes %d or 0 (unknown) but was %d" % (
        len(self.classes), len(soft_gt))

        if len(soft_gt) == 0:
            hard_gt = -1
            hard_gt_verbose = 'unknown'
        else:
            hard_gt = soft_gt.index(max(soft_gt))
            hard_gt_verbose = self.classes[hard_gt]


        if info is None:
            info = {}

        entry = {"path": path, "split": split, "gt_verbose": hard_gt_verbose, 'gt': hard_gt, 'soft_gt':soft_gt, 'info': info}
        return path, entry

    def add_image(self, path, split,  soft_gt: List[float], info=None):

        path, entry = self.add_helper(path,split,soft_gt,info)
        id = self.extract_unique_id_from_path(path)
        assert id not in self.images_dict, "You tried to add an already existing image to the dataset. Use update_image(), if you want to update the image instead."
        self.images_dict[id] = entry

    def update_image(self, path, split,  soft_gt: List[float], info=None):
        path, entry = self.add_helper(path, split, soft_gt, info)
        id = self.extract_unique_id_from_path(path)
        assert id in self.images_dict, "You tried to update an notexisting image to the dataset. Use add_image(), if you want to add an image."
        # ensure to keep old path
        entry['path'] = self.images_dict[id]['path']
        self.images_dict[id] = entry


    def get_image_iterator(self) -> Iterator[Tuple[str, str, str,List[float]]]:
        """
        Create an iterator for the images in this dataset,
        :return: iterator of tuples path (beginning with dataset name), dataset split and gt class (must be in classes)
        """
        for img_entry in self.images_dict.values():
            yield (img_entry['path'], img_entry['split'], img_entry['gt'], img_entry['soft_gt'])

    # def get_infos(self, path):
    #     """
    #     get complete info dictionary
    #     :param path:
    #     :return:
    #     """
    #     return self.images_dict[path]['info']

    def get(self,path, key):
        """
        try to get element based on path and potentially info
        :param path:
        :param key:
        :return: None if not found
        """
        id = self.extract_unique_id_from_path(path)
        entry = self.images_dict.get(id, None)

        if entry is None:
            # found no image to id / path
            return entry

        v = entry.get(key, None)

        if v is None:
            # found key not in direct dictionary -> search in info
            if "info" in entry:
                v = entry['info'].get(key, None)

        return v


    def get_number_images(self):
        return len(self.images_dict)

    def save_jsons(self, out_path : str):
        """
        save the dataset
        :param out_path:
        :return:
        """

        dataset_json = {"name": self.dataset_name, "v_fold": self.v_fold, "classes": self.classes, "budget": self.budget, "weighted_budget":self.weighted_budget, "images": list(self.images_dict.values())}

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # print("dump to file...")
        # print(len(self.images))
        with open(out_path, 'w') as outfile:
            json.dump(dataset_json, outfile)


    def extract_unique_id_from_path(self, path):
        """
        internal method, should not be used without necessary understanding,
        extract the unique id from a path, this unqiue id is used for referencing and hashing
        :param path:
        :return:
        """

        # check if already unique
        if ".png" in path:
            id = path.split("/")[-1][:-4] # take last element after / and remove .png
        else:
            id = path

        if "-[" in id:
            id = id.split("-[")[0] # legacy remove visualization elements

        return id

    def get_budget(self):
        return self.budget, self.weighted_budget



    def get_training_subsets(self, _split,  mode='soft', also_gt=False):
        """
        Get a subset of the data
        :param dataset_json:
        :param split: the desired data split or all
        :param mode specifies if soft or hard version of gt is used
        :param also_gt: boolean to indicate apply split (all) also to gt, might brake due to missing values
        :return:
        """

        # filter for split and unknown gt class
        paths = np.array([
            path
            for i, (path, split, _, _) in enumerate(self.get_image_iterator())
            if split == _split or _split == 'all'
        ])

        if mode == 'soft':
            gt = np.array([
                soft_gt
                for i, (_, split, hard_gt, soft_gt) in enumerate(self.get_image_iterator())
                if split == _split  or (also_gt and _split == 'all')# soft gt does not exists for all
            ])
        else:
            # hard mode
            num_classes = len(self.classes)
            gt = np.array([
                [1 if hard_gt == i else 0 for i in range(num_classes)]
                for i, (path, split, hard_gt, soft_gt) in enumerate(self.get_image_iterator())
                if split == _split or (also_gt and _split == 'all') # soft gt does not exists for all
            ])

        # detect erronous sets, sum needs to be one
        if len(gt) > 0:
            sums = np.sum(gt,axis=1, keepdims=True)
            # print(sums[:3],np.any(np.abs(sums-1)>0.01))
            # print(_split,gt[:3,:])

            if np.any(np.abs(sums-1)>0.01):
                # sum does not add up to one
                gt = gt/sums
                # print(gt[:3,:])

        # shuffle dataset once because datasets are structured
        indices = np.arange(paths.shape[0])
        random.seed(42)
        random.shuffle(indices)

        if len(paths) == len(gt):
            gt = gt[indices]
        else:
            print("WARNING: demanded soft gt but size did not match the paths, return empty set for split ", _split)
            gt = np.array([])
        paths = paths[indices]


        return paths, gt



