from random import Random

from absl import app
from absl import flags
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
import os
from os.path import join
import numpy as np

from src.algorithms.vps_files.proposals import proposals, class_blending
from src.util.oracle import AnnotationJson

FLAGS = flags.FLAGS
flags.DEFINE_list(name='tags',
                  help='Defines the tags which should be checked for the loaded annotations', default=[])

flags.DEFINE_float(name='dataset_coefficient', help="Enables the corrections based on the cleverlabel paper, called delta in paper, 0.1 is default and 0 meanse deactivated", default=0.1)
flags.DEFINE_float(name='blending_coefficient', help="Blending of the proposals and the estimated class distribution, called mu in the paper, 1 means only proposals (ingoring the simulation), 0 means only distribution, -1 means deactivated", default=0.75)

class VPSImport(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self,f'vps_{FLAGS.tags[0]}_{FLAGS.dataset_coefficient:0.02f}_{FLAGS.blending_coefficient:0.02f}')

        self.mode = 'soft'
        self.r = Random()
        self.r.seed(49312)

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):
        """
            load the annotated data based on the user study
            :param ds:
            :param oracle:
            :param dataset_info:
            :param v_fold:
            :param num_annos: number of used annotations, -10 means all annotations without sampling
            :param percentage_labeled: ignored in this implmentation, maybe use as proxy for repetitions
            :return:
            """


        tags = FLAGS.tags

        if len(tags) == 1 and os.path.exists(join(f"src/algorithms/vps_files/{tags[0]}.json")):
            annotations = AnnotationJson.from_file(f"src/algorithms/vps_files/{tags[0]}.json") # load from manually stored cache

        else:
            # manual annotations not provided thus this loading will fail
            # add annotations based on tags
            annos = [anno for anno in os.listdir(join(dataset_info.root_directory, "manual_annotations", "vps")) if
                     any([t in anno for t in tags])]

            print("Imported annotations", annos)
            annotations = AnnotationJson.empty(self.name)
            for anno in annos:
                annotations.add_other_anno_file(join(dataset_info.root_directory, "manual_annotations", "vps", anno),
                                                ignore_wrong_dataset=True, ignore_class=['unknown'], check_for_annonimous=False)


            annotations.anoymize()
            # manual saving of configurations
            annotations.save_json_list("src/algorithms/vps_files")

        imgs, classes, data = annotations.get_probability_data(return_numbers=True)

        # rearrange classes to be equal to dataset info classes
        data = data.transpose()
        data = np.array([data[classes.index(cl)] for cl in dataset_info.classes])
        data = data.transpose()
        classes = dataset_info.classes # order of classes now changed

        # need to map old file names to new ones in ds
        for i, old_path in enumerate(imgs):
            id = old_path.split("/")[-1].split("-") # remove wrong elements of file names
            id = id[0] + "-" + id[1]

            # get distribution of annotations
            distribution = data[i] / np.sum(data[i])

            verbose = False

            if verbose:
                print(id)
                print("proposed class ", proposals[id])
                print("GT distribution: ", oracle.get_soft_gt(id,-1))
                print("Vps distribution, ", distribution)



            #sample from this distribution as specified
            anno = np.zeros((len(distribution)))
            if num_annos > 0:
                labels = self.r.choices(range(len(distribution)), weights=distribution, k=num_annos)
                for l in labels:
                    anno[l] += 1
            elif num_annos == -10:
                # use all annotations
                anno = data[i]

            used_num_annos = np.sum(anno)

            calculated_distribution =  anno / np.sum(anno)

            if verbose:
                print("calculated distribution: ", calculated_distribution)

            # maybe correct it
            if FLAGS.dataset_coefficient > 0:
                calculated_distribution = bias_correction(anno,FLAGS.dataset_coefficient,used_num_annos,proposals[id])

            if verbose:
                print("corrected distribution: ", calculated_distribution)

            # maybe blend it
            if FLAGS.blending_coefficient >= 0:
                t = FLAGS.blending_coefficient
                calculated_distribution = t * (calculated_distribution) +\
                                          (1 - t) * class_blending[
                                              np.argmax(calculated_distribution) if used_num_annos > 0 else proposals[id]
                                          ]

            if verbose:
                print("blended distribution: ", calculated_distribution)

            split = ds.get(id, 'original_split')  # determine original split before move to unlabeled
            ds.update_image(id, split, list(calculated_distribution)) # inherentily casts to identifier

     
        return ds


def bias_correction(annos, offset, num_annos, proposed_class):
    """

    :param annos: annotations for distribution calculation
    :param offset: used offset for confimration bias correction
    :param num_annos: number of annotations
    :param proposed_class: the proposed class
    :return:
    """
    # need to lower this score based on confirmation offset
    pc = annos[proposed_class] / num_annos
    corrected = (pc - offset) / (0.99 - offset)
    corrected = min(1, max(0, corrected))

    m = max(1, num_annos - annos[proposed_class])
    p = annos / m

    p *= (1 - corrected)  # rescale with leftovers
    p[proposed_class] = corrected

    return p

def kl_div(y_true, y_pred, epsilon = 0.00001):
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

def main(argv):
    """
       Apply only initial annotation
       :return:
    """

    alg = VPSImport()
    alg.apply_algorithm()

    alg.report.show()





if __name__ == '__main__':
    app.run(main)
