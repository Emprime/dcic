import os
import pickle
from os.path import join
from itertools import product
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson

NUM_MAX_SLICES = 5 # can only be 5 due to five folds, often it should be 3

class DCICReport:

    def __init__(self):
        self.scores = {}

        self.init_new_experiment()

    def init_new_experiment(self):
        """
        Initialize the new experiment run, reset all major logging of internal metrics
        :return:
        """

        self.dataset_jsons = []
        self.y_trues = []
        self.y_preds = []
        # current scores for experiment
        self.cs = {prefix+k:[] for prefix in ['','input_'] for k in ['macro_f1','macro_acc', 'kl', 'ece'] } #'c_matrix', 'report',
        self.cs['input_consistency'] = {}
        self.cs['budget'] = []
        self.cs['w_budget'] = []
        self.cl = None

        self.show_score_names = ['kl','input_kl','budget','macro_acc','input_macro_acc']

    def ece(self,y_true, y_pred, num_bins=10):
        """
        Implementation inspired by https://lars76.github.io/2020/08/07/metrics-for-uncertainty-estimation.html
        :param y_true:
        :param y_pred:
        :param num_bins:
        :return:
        """
        pred_y = np.argmax(y_pred, axis=-1)
        correct = (pred_y == y_true).astype(np.float32)
        prob_y = np.max(y_pred, axis=-1)

        b = np.linspace(start=0, stop=1.0, num=num_bins)
        bins = np.digitize(prob_y, bins=b, right=True)

        o = 0
        for b in range(num_bins):
            mask = bins == b
            if np.any(mask):
                o += np.abs(np.sum(correct[mask] - prob_y[mask]))

        return o / y_pred.shape[0]

    def kl_div(self,y_true, y_pred, epsilon = 0.00001):
        """
           own kullback  loss implementation
           :param y_true:
           :param y_pred:
           :return:
        """
        # seems to be identical to keras loss
        y = np.clip(y_true, epsilon, 1 - epsilon)
        x = np.clip(y_pred, epsilon, 1 - epsilon)

        kl = np.sum(y * np.log(y / x), axis=1)

        return np.mean(kl)

    def inner_score_calcualtion(self,y_true,y_pred, verbose, prefix=""):

        # print(y_true.shape,y_pred.shape)

        # manually caluclate kl divergence
        # m = tf.keras.metrics.KLDivergence()
        # m.update_state(y_true, y_pred)
        # kl = m.result().numpy()

        # print(kl)

        kl = self.kl_div(y_true, y_pred)
        # print(kl)

        # cast to hard labels
        y_true = np.argmax(y_true, axis=1)

        # calculate ece
        ece = self.ece(y_true,y_pred)

        # cast to hard labels
        y_pred = np.argmax(y_pred, axis=1)

        self.cs[prefix + 'ece'].append(ece)

        if verbose > 1:
            print("## " + prefix + "scores ##")
            print("KL-divergence: %0.04f" % kl)
            print("ECE: %0.04f" % ece)

        c_matrix = confusion_matrix(y_true, y_pred)
        if verbose > 1:
            print("Confusion matrix: rows actual, columns prediction")
            print(c_matrix)
        c_matrix_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        if verbose > 1:
            print(classification_report(y_true, y_pred, digits=4, target_names=self.cl, zero_division=0))
        cl_report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)

        assert len(self.cl) == len(c_matrix), F"Dimensions do not fit {len(self.cl)} vs. {len(c_matrix)}"

        f1 = cl_report['macro avg']['f1-score']
        acc = np.trace(c_matrix_normalized) / len(self.cl)

        # save values
        # self.cs[prefix + 'c_matrix'].append(c_matrix_normalized)
        self.cs[prefix +'kl'].append(kl)
        # self.cs[prefix +'report'].append(cl_report)
        self.cs[prefix +'macro_f1'].append(f1)
        self.cs[prefix +'macro_acc'].append(acc)

        return f1, acc, kl

    def _calculate_scores(self, verbose):
        """
        internal method to calculate the scores based on the internal logged predictions
        :return:
        """

        # init
        f1, acc, kl, cl_report, c_matrix_normalized = -1, -1, -1, None, None

        # calculate score fur currently logged elements if not already done
        for i, (dataset_json, y_true, y_pred) in enumerate(zip(self.dataset_jsons, self.y_trues, self.y_preds)):
            if i < len(self.cs.get("kl",[])):
                continue # scores have already been calculated

            # print(dataset_json.dataset_name)
            # print(y_true.shape)
            # print(y_pred.shape)

            # store classes
            cl = dataset_json.classes
            assert self.cl is None or self.cl == cl
            self.cl = cl

            # extract inputs for consistency
            for path, split, hard_gt, soft_gt in dataset_json.get_image_iterator():
                temp = self.cs['input_consistency'].get(path, [])
                if len(soft_gt) > 0:  # check soft gt is known
                    temp.append(np.argmax(soft_gt))
                else:
                    temp.append(-1)
                self.cs['input_consistency'][path] = temp

            # extract budgets
            num_items_not_test = len([1 for _, split, _, _ in dataset_json.get_image_iterator() if split != 'test'])
            budget = dataset_json.budget / float(num_items_not_test)
            w_budget = dataset_json.weighted_budget / float(num_items_not_test)
            self.cs['budget'].append(budget)
            self.cs['w_budget'].append(w_budget)

            # analyse input elements
            dataset_name = dataset_json.dataset_name
            dataset_info = get_all_dataset_infos()[dataset_name]
            split_number = i+1 if "verse" not in dataset_name else 3
            org_dataset_json = DatasetDCICJson.from_file(join(dataset_info.evaluate_directory,
                                                              "{name}-slice{split}.json".format(name=dataset_name,
                                                                                                split=split_number)))


            # print(len(org_dataset_json))
            # print(len(dataset_json))

            provided_gt = np.array([
                soft_gt
                for i, (_, split, hard_gt, soft_gt) in enumerate(dataset_json.get_image_iterator())
                if split == 'test'
            ])

            org_gt = np.array([
                soft_gt
                for i, (_, split, hard_gt, soft_gt) in enumerate(org_dataset_json.get_image_iterator())
                if split == 'test'
            ])

            if verbose > 1:
                print("# Calculation of scores for fold", i)

            # print(provided_gt.shape)
            # print(org_gt.shape)
            # print(org_gt)


            # calculate scores for trained and input values
            if y_true is not None and y_pred is not None:
                f1, acc, kl = self.inner_score_calcualtion(y_true,y_pred,verbose)
            else:
                f1, acc, kl = -1, -1, -1

            # print(org_gt.shape, provided_gt.shape)
            if org_gt.shape[1] > 0 and provided_gt.shape[1] > 0 and len(org_gt) == len(provided_gt): # check data is available and of same size
                self.inner_score_calcualtion(org_gt, provided_gt, verbose, prefix="input_")


        # return last values which were calculated
        return f1, acc, kl


    def end_run(self, dataset_json, y_true, y_pred, verbose=1):
        """
        Calculate scores at the end of a fold analysis run
        :param y_true:
        :param y_pred:
        :return:
        """

        # log results to intermediate lists
        self.dataset_jsons.append(dataset_json)
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)


        f1, acc, kl = self._calculate_scores(verbose)

        return f1, acc, kl


    def _calculate_values_from_scores(self,k,v, verbose):

      # assert self.cl is not None, "Classes are missing, you need to successfully call end_run() before this method"



        if k == 'input_consistency':

            annotations = np.array([v[key] for key in v]).astype(int)
            if len(annotations) == 0:
                print("Missing data")
                return np.array([-1])
            num_classes = np.amax(annotations)+1
            num_splits = len(annotations[0])
            consistency_matrix = np.zeros((num_splits, num_splits))
            ignore_unknown_consistency_matrix = np.zeros((num_splits, num_splits))
            # calculate consistencies
            for i in range(num_splits):
                for j in range(num_splits):
                    if i < j:
                        # calculate score
                        kappa = cohen_kappa_score(annotations[:, i], annotations[:, j])
                        # print(annotations[:10, i], annotations[:10, j] )
                        try:

                            ignore_unknown_class_kappa = cohen_kappa_score(annotations[:, i], annotations[:, j],
                                                                           labels=np.arange(num_classes))
                        except ValueError:
                            # generated data is maybe to bad and does not contain all the values
                            print("bad predictions")
                            ignore_unknown_class_kappa = np.NaN


                        consistency_matrix[i, j] = kappa
                        ignore_unknown_consistency_matrix[i, j] = ignore_unknown_class_kappa
                        consistency_matrix[j, i] = kappa
                        ignore_unknown_consistency_matrix[j, i] = ignore_unknown_class_kappa

                    elif i == j:
                        # trivial
                        consistency_matrix[i, i] = 1
                        ignore_unknown_consistency_matrix[i, i] = 1

            if verbose > 4:
                print("Consistency matrix of input")
                print(consistency_matrix)
                print("Consistency matrix of input (igoring unknown class)")
                print(ignore_unknown_consistency_matrix)

            values = np.array(
                [ignore_unknown_consistency_matrix[i, j] for i, j in product(range(num_splits), range(num_splits)) if i < j])
        else:
            values = np.array(v)
        return values

    def summarize_and_reset(self, experiment, mode, save=False, verbose=1):
        """
        Show detailed results over the current experiment and log the results to an undeleteable log and reset after this
        :return:
        """


        # save the current lists to file system
        if save:
            self._save_internal_elements(experiment,mode)


        # calculate summary over all splits
        if verbose > 0:
            print(f"###### {experiment} with mode {mode} ########")
            print("################ Scores ####################")
            for k in self.show_score_names:
                v = self.cs[k]
                values = self._calculate_values_from_scores(k,v, verbose)

                print("%s: %0.04f +- %0.04f" % (k, np.mean(values), np.std(values)))
            print("###########################################")


        # reset
        self.scores[experiment] = self.cs
        self.init_new_experiment()


    def show(self):
        """
        Show the complete result table up to now
        :return:
        """

        # create header
        first_padding = 25
        padding = 19
        keys = self.show_score_names
        result_table = "Experiment".ljust(first_padding)
        for k in keys:
            result_table +=  " | " + k.ljust(padding)
        result_table += "\n"
        result_table += "   ---".ljust(first_padding)
        for k in keys:
            result_table += " | " + "---".ljust(padding)
        result_table += "\n"

        # create rows
        for experiment, cs in self.scores.items():
            # print(experiment)
            result_table += experiment.ljust(first_padding)
            for k in self.show_score_names:
                v = cs[k]
                # print(k,v)
                values = self._calculate_values_from_scores(k,v,verbose=0)
                # print(values)
                result_table += " | " + f"{np.mean(values):0.04f} +- {np.std(values):0.04}".ljust(padding)
            result_table += "\n"

        print(result_table)


    def load_results(self,use_cache=True):
        """
        load the given results
        :param use_cache: determines if a cache is used for the scores
        :return:
        """

        directory = "/data/evaluation_logs"
        cache_file = join(directory,"cache.pkl")
        if use_cache and os.path.exists(cache_file):
            print("Loading cache ...")
            with open(cache_file, 'rb') as handle:
                self.scores = pickle.load(handle)
        else:
            # load one folder after the other, each folder is interpreted as one experiment
            for experiment in sorted([folder for folder in os.listdir(directory) if os.path.isdir(join(directory,folder))]):
                print("Load results of ", experiment)

                # if not "verse" in experiment:
                #     continue

                files = sorted(os.listdir(join(directory,experiment)))
                # load elements
                for file in files:
                    if file.startswith("data"):
                        j1 = DatasetDCICJson.from_file(join(directory,experiment, file))
                        # check for potential newer datasets
                        try:
                            temp = sorted(os.listdir(join("/data/output_datasets",experiment)))
                            j2 = DatasetDCICJson.from_file(join("/data/output_datasets",
                                                               experiment,temp[int(file.split("_")[1][0])]
                                                               ))

                        except:
                            # ignore
                            j2 = None
                        # print(j2, len(j2) if j2 is not None else "MIssing", len(j1))
                        if j2 is None:
                            j = j1
                        else:
                            # print("Used newer data json 2")
                            j = j2

                        self.dataset_jsons.append(j)
                    elif file.startswith("yp"):
                        yp = np.loadtxt(join(directory,experiment, file), delimiter=',')
                        self.y_preds.append(yp)
                    elif file.startswith("yt"):
                        yt = np.loadtxt(join(directory, experiment, file), delimiter=',')
                        # print(file,len(yt))
                        self.y_trues.append(yt)

                # check if entries are valid
                valid = True
                if len(self.y_trues) > NUM_MAX_SLICES or len(self.y_trues) == 0 or len(self.y_preds) != len(self.y_trues) or len(self.dataset_jsons) != len(self.y_trues):
                    print(f"ERROR: {experiment} the number of loaded files exceeds maximum (5), are not available or are not of the same amount. Please check the evaluation logs")
                    valid = False

                # check that sizes are the same
                for i in range(len(self.y_trues)):
                    if len(self.y_preds[i]) != len(self.y_trues[i]):
                        valid = False
                        print(f"Error: {experiment} found incompatible sizes"
                              f" of loaded data {len(self.y_preds[i])} vs. {len(self.y_trues[i])}")

                        print(f"Error: {experiment} found incompatible sizes"
                              f" of loaded data {self.y_preds[i][:3]} vs. {self.y_trues[i][:3]}")
                        break



                if valid:
                    # calculate scores per folder
                    self._calculate_scores(verbose=0)

                    # save calculated scores
                    self.scores[experiment] = self.cs

                # reset
                self.init_new_experiment()


            # save to cache
            print("Creating cache ...")
            with open(cache_file, 'wb') as handle:
                pickle.dump(self.scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _save_internal_elements(self, experiment_name, mode):
        """
        save the results for one experiment as permanent file into the folder `evaluation_logs`
        :return:
        """

        path = join("/data/evaluation_logs", f"{experiment_name}")
        os.makedirs(path, exist_ok=True)
        # might not work with high parallelization
        # self.df.to_csv(join(path, f"logs_{time.strftime('%d-%m-%Y-%H-%M-%S')}.csv"))

        # overwrite elements
        for i, j in enumerate(self.dataset_jsons):
            j.save_jsons(join(path,"data_%d.json" % i))

        for i, y_p in enumerate(self.y_preds):
            np.savetxt(join(path,"yp_%d.csv" % i), y_p, delimiter=',', fmt="%0.5f")

        for i, y_t in enumerate(self.y_trues):
            np.savetxt(join(path,"yt_%d.csv" % i), y_t, delimiter=',', fmt="%0.5f")
