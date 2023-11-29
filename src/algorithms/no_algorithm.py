from absl import app
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton


class No_algorithm(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self,'no_alg')

        # ACTIVATE EVALUATION OPTION
        # DO NOT USE THIS OPTION EXCEPT FOR THIS CLASS
        self.cheating = True

        self.mode = 'soft'

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):

        paths, targets = ds.get_training_subsets('all')


        # reiterate over all elements and reassign them the desired value
        print(len(paths))
        for i, path in enumerate(paths):
            if i < 10:
                print(path, ds.get(path, 'soft_gt'))
        return ds

def main(argv):
    """
       Apply only initial annotation
       :return:
    """

    alg = No_algorithm()
    alg.apply_algorithm()

    alg.report.show()



if __name__ == '__main__':
    app.run(main)
