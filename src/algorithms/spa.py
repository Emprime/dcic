import glob
import traceback
from os.path import join
from absl import app
import numpy as np
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from absl import flags
from src.util.const import get_all_dataset_infos
from src.util.json import DatasetDCICJson
from src.util.oracle import AnnotationOracle

FLAGS = flags.FLAGS

flags.DEFINE_list(name='folders',
                  help='the folders which should be evaluated', default=[])

# SPA specific parameters
flags.DEFINE_bool(name='enable_inclusion_label', help="defines if the original training labels should be blended", default=False)
flags.DEFINE_bool(name='enable_offset_correction', help="defines if a correction of the simulation baed on the proposal offset should be calculated", default=False)
flags.DEFINE_bool(name='enable_annotate_unlabeled_data', help="defines if unlabeled data should be annotated", default=False)
flags.DEFINE_integer(name='simulation_repetition', help="Number of repetitions of the simulation", default=1)
flags.DEFINE_float(name='blending_coefficient', help="Blending of the proposals and the estimated class distribution, 1 means only proposals (ingoring the simulation), 0 means only distribution", default=0.75)


# analysed values based on slope analysis
dataset_proposal_acceptance_offsets = {
    'Plankton':0.6481, 'MiceBone': 0.4103, 'Turkey':0.1417, 'CIFAR10H':0.0,
    'QualityMRI': 0, 'Benthic': 0.4017, 'Treeversity#1': 0.2608, 'Treeversity#6': 0.2067,
    'Pig': 0.2572, 'Synthetic': 0.2608
}

dataset_approximated_proposal_acceptance_offsets = {
    'Plankton':0.1, 'MiceBone': 0.1, 'Turkey':0.1, 'CIFAR10H':0.1,
    'QualityMRI': 0.1, 'Benthic': 0.1, 'Treeversity#1': 0.1, 'Treeversity#6': 0.1,
    'Pig': 0.1, 'Synthetic': 0.1
}


dataset_class_approximations = {
    # KL 0.4286257124664755
    'Benthic': {
        'coral':        [0.814, 0.000, 0.000, 0.057, 0.114, 0.014, 0.000, 0.000],
        'crustacean':   [0.043, 0.843, 0.000, 0.000, 0.114, 0.000, 0.000, 0.000],
        'cucumber':     [0.000, 0.000, 0.900, 0.000, 0.100, 0.000, 0.000, 0.000],
        'encrusting':   [0.024, 0.000, 0.000, 0.756, 0.040, 0.052, 0.000, 0.128],
        'other_fauna':  [0.021, 0.016, 0.000, 0.037, 0.805, 0.042, 0.000, 0.079],
        'sponge':       [0.000, 0.019, 0.000, 0.062, 0.044, 0.844, 0.025, 0.006],
        'star':         [0.000, 0.000, 0.000, 0.000, 0.030, 0.000, 0.970, 0.000],
        'worm':         [0.017, 0.000, 0.017, 0.075, 0.058, 0.000, 0.000, 0.833]
    },

    # KL:  0.18304355047068008
    'CIFAR10H':  {
        'airplane':     [0.950, 0.000, 0.000, 0.013, 0.000, 0.000, 0.000, 0.000, 0.037, 0.000],
        'automobile':   [0.000, 0.978, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.022],
        'bird':         [0.000, 0.000, 0.925, 0.008, 0.033, 0.017, 0.000, 0.017, 0.000, 0.000],
        'cat':          [0.000, 0.000, 0.008, 0.875, 0.017, 0.042, 0.042, 0.008, 0.000, 0.008],
        'deer':         [0.000, 0.000, 0.000, 0.000, 0.929, 0.036, 0.000, 0.036, 0.000, 0.000],
        'dog':          [0.000, 0.000, 0.000, 0.044, 0.000, 0.956, 0.000, 0.000, 0.000, 0.000],
        'frog':         [0.000, 0.000, 0.017, 0.008, 0.000, 0.000, 0.975, 0.000, 0.000, 0.000],
        'horse':        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
        'ship':         [0.008, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.977, 0.015],
        'truck':        [0.014, 0.029, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.957]},

    # KL:  0.1822585877541493
    'MiceBone': {
        'g':  [0.727, 0.180, 0.093],
        'nr': [0.033, 0.868, 0.099],
        'ug': [0.06, 0.167, 0.773]},

    # KL:  0.28142176337024305
    'Plankton': {
        'bubbles':                  [0.950, 0.000, 0.000, 0.000, 0.000, 0.050, 0.000, 0.000, 0.000, 0.000],
        'collodaria_black':         [0.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        'collodaria_globule':       [0.000, 0.033, 0.900, 0.000, 0.000, 0.067, 0.000, 0.000, 0.000, 0.000],
        'cop':                      [0.000, 0.000, 0.000, 0.911, 0.000, 0.033, 0.000, 0.000, 0.000, 0.056],
        'det':                      [0.033, 0.011, 0.000, 0.000, 0.800, 0.111, 0.000, 0.044, 0.000, 0.000],
        'no_fit':                   [0.000, 0.000, 0.000, 0.015, 0.021, 0.941, 0.000, 0.003, 0.009, 0.012],
        'phyto_puff':               [0.000, 0.000, 0.000, 0.000, 0.025, 0.000, 0.900, 0.075, 0.000, 0.000],
        'phyto_tuft':               [0.013, 0.000, 0.000, 0.000, 0.031, 0.000, 0.006, 0.950, 0.000, 0.000],
        'pro_rhizaria_phaeodaria':  [0.000, 0.025, 0.008, 0.000, 0.008, 0.033, 0.000, 0.000, 0.925, 0.000],
        'shrimp':                   [0.000, 0.000, 0.000, 0.000, 0.017, 0.100, 0.000, 0.000, 0.000, 0.883]},

    # KL: 0.3932698040768524
    'Synthetic': {
        'bc': [0.845, 0.075, 0.025, 0.025, 0.030, 0.000],
        'be': [0.100, 0.767, 0.017, 0.050, 0.000, 0.067],
        'gc': [0.052, 0.012, 0.756, 0.104, 0.052, 0.024],
        'ge': [0.011, 0.061, 0.106, 0.761, 0.011, 0.050],
        'rc': [0.018, 0.018, 0.065, 0.006, 0.788, 0.106],
        're': [0.036, 0.021, 0.029, 0.029, 0.079, 0.807]},

    # KL: 0.2856125819585248
    'Pig':  {
        '1_intact':         [0.671, 0.160, 0.097, 0.071],
        '2_short':          [0.123, 0.650, 0.131, 0.096],
        '3_fresh':          [0.067, 0.153, 0.683, 0.097],
        '4_notVisible':     [0.067, 0.156, 0.156, 0.622]
    },

    # KL: 0.29562331399407754
    'Treeversity#1':  {
        'bark':         [0.930, 0.000, 0.000, 0.000, 0.060, 0.010],
        'bud':          [0.005, 0.819, 0.067, 0.067, 0.038, 0.005],
        'flower':       [0.000, 0.034, 0.883, 0.046, 0.031, 0.006],
        'fruit':        [0.000, 0.050, 0.036, 0.836, 0.064, 0.014],
        'leaf':         [0.000, 0.033, 0.000, 0.033, 0.889, 0.044],
        'whole_plant':  [0.009, 0.009, 0.018, 0.000, 0.018, 0.945]
    },

    # KL: 0.34350036180164795
    'Treeversity#6':  {
        'bark':         [0.812, 0.025, 0.025, 0.000, 0.087, 0.050],
        'bud':          [0.130, 0.610, 0.075, 0.035, 0.150, 0.000],
        'flower':       [0.024, 0.110, 0.741, 0.014, 0.093, 0.017],
        'fruit':        [0.033, 0.042, 0.050, 0.700, 0.175, 0.000],
        'leaf':         [0.040, 0.104, 0.084, 0.060, 0.688, 0.024],
        'whole_plant':  [0.100, 0.000, 0.050, 0.017, 0.133, 0.700]
    },

    # KL: 0.04067997329557361
    'QualityMRI': {
        '0': [0.696, 0.304],
        '1': [0.240, 0.760]},

    # KL: 0.13049150048780664
    'Turkey': {
        'head_injury':      [0.833, 0.047, 0.120],
        'not_injured':      [0.060, 0.780, 0.160],
        'plumage_injury':   [0.012, 0.039, 0.949]},
}

class SimulatedProposalAcceptance(AlgorithmSkelton):
    """
    Class to import manual predictions from other repositories for the evaluation
    """

    def __init__(self, raw_name, report=None):
        """
                Initializes an instance of the SimulatedProposalAcceptance class.

                :param raw_name: The raw name of the proposal.
                :param report: Optional report object to store results.
        """

        self.raw_name = raw_name
        original_method = raw_name.split("-")[1]

        name = original_method + f"_spa{FLAGS.simulation_repetition:03d}_bl{FLAGS.blending_coefficient:0.02f}"

        if FLAGS.enable_offset_correction:
            expected_dataset_name = self.raw_name.split("-")[0]
            name += f"_off_correct{dataset_approximated_proposal_acceptance_offsets[expected_dataset_name]:0.04f}"

        if FLAGS.enable_annotate_unlabeled_data:
            name += "_inc_un"

        if FLAGS.enable_inclusion_label:
            name += "_inc_lab"


        AlgorithmSkelton.__init__(self, name)

        self.proposed_ds = None

        if report is not None:
            self.report = report


    def apply_algorithm(self):
        """
        Override main method to allow the estimation only based on the folder name
        :return:
        """

        # determine percentage labeled and number annos based on folder name
        expected_dataset_name = self.raw_name.split("-")[0]
        dataset_info = get_all_dataset_infos()[expected_dataset_name]
        num_annos,percentage_labeled = self.raw_name.split("-")[-2:]
        num_annos = int(num_annos)
        percentage_labeled = float(percentage_labeled)

        oracle = AnnotationOracle(join(dataset_info.raw_data_directory, 'annotations.json'))


        # iterate over slices
        for v_fold in self.slices:

            file_name = f'{self.raw_name}-{v_fold}.json'

            print(f"# Start processing {file_name}")
            # load proposed dataset
            self.proposed_ds = DatasetDCICJson.from_file(join(dataset_info.root_directory,FLAGS.output_folder, self.raw_name,file_name))
            dataset_name = self.proposed_ds.dataset_name
            assert expected_dataset_name == dataset_name, f"Mismatch between folder name {expected_dataset_name} and dataset name {dataset_name}"
            assert dataset_info.name == dataset_name, f"Mismatch between loaded dataset info name {dataset_info.name} and dataset name {dataset_name}"


            # setup dataset
            raw_ds = DatasetDCICJson.from_file(join(dataset_info.input_directory, f'{dataset_name}-slice{v_fold}.json'))

            # setup oracle and relabeled dataets
            new_ds = DatasetDCICJson.from_definition(dataset_name, v_fold, dataset_info.classes)
            oracle.init(new_ds)

            # ensure that all underling distributions are of the same classes
            assert oracle.get_classes() == raw_ds.classes
            assert oracle.get_classes() == new_ds.classes
            assert oracle.get_classes() == dataset_info.classes

            # Get initial annotations

            print(f"# Get initial annotations for  {file_name}")
            new_ds = self.get_initial_annotations(raw_ds, new_ds, oracle,dataset_info,v_fold, num_annos, percentage_labeled)

            # Apply main algorithm
            print(f"Apply {self.name} to {file_name} with {num_annos} annotations on {percentage_labeled} of the data")
            try:
                new_ds = self.run(new_ds,oracle, dataset_info,v_fold,num_annos,percentage_labeled)
            except Exception as e:
                print(e)
                traceback.print_exc()
                new_ds = None

            if new_ds is None:
                continue


            self.after_run_finished(new_ds)


            b = new_ds.budget
            new_folder =  f"{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}"
            new_file = f'{dataset_name}-{self.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}.json'

            new_ds.save_jsons(join(dataset_info.root_directory, FLAGS.output_folder, new_folder, new_file))

        self.after_all_slices_processed(dataset_info.name, percentage_labeled, num_annos)

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):
        """
        Runs the algorithm on a dataset.

        1. It initializes various variables and retrieves dataset information.
        2. It iterates over the proposed dataset, processing each image.
        3. For each image:
            - Retrieves the original split and predicted probabilities.
            - Determines if unlabeled data should be annotated based on FLAGS.enable_annotate_unlabeled_data or the original split should be kept
            - Simulates proposal acceptance using simulate_proposal_acceptance
            - Updates the dataset with the estimated distribution.
        4. Prints debug information about the final budget and KL divergence.
        5. Returns the updated dataset.

        Global variables starting with FLAGS that are used in this method include:

        - FLAGS.simulation_repetition: Determines the number of simulation repetitions aka the number of simulated annotations per image
        - FLAGS.enable_annotate_unlabeled_data: Controls whether unlabeled data should be annotated, otherwise simulation is only executed on previously labeled data during the initiliazation
        - FLAGS.enable_offset_correction: Indicates whether offset correction should be applied. (called BC in paper)
        - FLAGS.blending_coefficient: Controls the blending coefficient used in updating the dataset, called CB in paper and this parameter is mu
        - FLAGS.enable_inclusion_label: Determines whether inclusion labels are enabled, it enhances the distribution with the given labels from the initilization

        :param ds: The dataset to run the algorithm on.
        :param oracle: The AnnotationOracle object for annotation management.
        :param dataset_info: The information about the dataset like number of classes and class labels
        :param v_fold: The fold index of the dataset.
        :param num_annos: The number of annotations which were initialized
        :param percentage_labeled: The percentage of labeled data which were initialized

        :return: The modified dataset after running the algorithm.
        """



        # determine the proposal offset acceptance for the simulation
        if dataset_info.name not in dataset_proposal_acceptance_offsets or dataset_info.name not in dataset_approximated_proposal_acceptance_offsets:
            print("WARNING: Missing proposal acceptance offset in 'predictions_import.py', Simulation not possbile")
            return None
        else:
            proposal_acceptance_offset = dataset_proposal_acceptance_offsets[dataset_info.name]
            proposal_acceptance_offset_approximated = dataset_approximated_proposal_acceptance_offsets[dataset_info.name]


            # check dataset classes
            averages_per_class = dataset_class_approximations[dataset_info.name]
            assert list(averages_per_class.keys()) == dataset_info.classes
            averages_per_class = np.array([averages_per_class[key] for key in averages_per_class])



        # iterate over proposed dataset
        for path, split, _, _ in self.proposed_ds.get_image_iterator():
            original_split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
            predicted_prob = self.proposed_ds.get(path,'soft_gt')


            proposed_class =  np.argmax(predicted_prob, axis=-1) # replace with predicted_prob for probabilistic proposals

            # check if unlabled data should be annotated
            if FLAGS.enable_annotate_unlabeled_data or\
                    original_split == ds.get(path,'split'):

                # simulated proposal acceptance
                estimated_distribution = \
                    self.simulate_proposal_acceptance(
                        proposed_class, oracle,
                        FLAGS.simulation_repetition, path, proposal_acceptance_offset,
                        correct_with_offset=proposal_acceptance_offset_approximated if FLAGS.enable_offset_correction else -1,
                        ignore_costs=original_split == 'test'
                    )

            else:
                estimated_distribution = predicted_prob

            # convert to numpy
            estimated_distribution = np.array(estimated_distribution)

            # expects the same unique file name in path like in the original
            t = FLAGS.blending_coefficient
            if FLAGS.enable_inclusion_label and ds.get(path, 'split') in ['train', 'val']:
                # get input estimated
                init_label = np.array(ds.get(path, 'soft_gt'))
                sr = FLAGS.simulation_repetition
                ds.update_image(path, original_split, list(
                    t * (
                            (sr / (sr + num_annos)) * estimated_distribution +
                            (num_annos / (sr + num_annos)) * init_label
                    )
                    +
                    (1 - t) * averages_per_class[np.argmax(estimated_distribution) if sr > 0 else proposed_class]
                ))
            else:
                ds.update_image(path, original_split, list(t * estimated_distribution + (1 - t) * averages_per_class[
                    np.argmax(estimated_distribution)]))

        # DEBUG INFORMATION
        num_items_not_test = float(len([1 for _, split, _, _ in ds.get_image_iterator() if split != 'test']))
        print(f"Final Budget: {ds.weighted_budget/num_items_not_test} / {ds.budget/num_items_not_test} [{num_items_not_test}]")

        # KL with regard to unknown gt
        print(f"Final KL:")
        for split in ['train','val','test','unlabeled']:
            paths, estimated_gts = ds.get_training_subsets(split)
            real_gt = []
            for path,estimated_gt in zip(paths,estimated_gts):
                soft_gt = oracle.get_soft_gt(path, -1)
                real_gt.append(soft_gt)

            real_gt, estimated_gts = np.array(real_gt), np.array(estimated_gts)
            # print(real_gt.shape,estimated_gts.shape)
            if len(real_gt) > 0 and len(estimated_gts[0]) > 0:
                print(f"{split} - {kl_div(real_gt,estimated_gts)} [{len(real_gt)}]")
            else:
                print(split, f"- found {len(real_gt)} Entries")

        return ds





    def simulate_proposal_acceptance(self,proposed_class, oracle, simulation_repetitions,path,proposal_acceptance_offset,correct_with_offset=-1, ignore_costs=False, simulated_label = None):
        """
            Simulates the proposal acceptance process and returns the estimated distribution.

            1. It retrieves the soft ground truth from the oracle.
            2. If ignore_costs is false, it increases annotation count in the oracle.
            3. It runs the simulation (SPA) multiple times:
                - Selects a class for the proposal.
                - Calculates the acceptance rate based on soft ground truth.
                - Determines if the proposal is accepted or selects another class.
                - Updates the simulated labels based on the simulation results.
            4. It returns the simulated labels or applies corrections (BC in paper) if specified.

            :param proposed_class: The proposed class for the simulation.
            :param oracle: The AnnotationOracle object for annotation management.
            :param simulation_repetitions: The number of simulation repetitions aka the number of simulated annotations
            :param path: The path of the current image.
            :param proposal_acceptance_offset: The offset value for proposal acceptance, delta in the paper for SPA
            :param correct_with_offset: The offset value for correction. Default is -1, delta in the paper for CleverLabel
            :param ignore_costs: Flag indicating whether to ignore annotation costs. Default is False.
            :param simulated_label: A given label distribution to prevent the simulation and only correct and improve the resuls. Default is None.

            :return: The estimated distribution after the simulation.
            """

        if isinstance(proposed_class, list):
            # workaround if a probability distribution is given instead of only one class
            soft_proposed_class = proposed_class
            soft_proposed_class= np.round(soft_proposed_class,decimals=4)
            # ensure sum to one, dirty workaournd because some last digitis might be cutted
            total = np.sum(soft_proposed_class)
            soft_proposed_class[np.argmax(soft_proposed_class)] += 1 - total
            proposed_class = np.argmax(proposed_class,axis=-1)
        else:
            soft_proposed_class = None

        # get soft gt
        soft_gt = oracle.get_soft_gt(path, -1)
        # increase annotation count appropriatly based on simulation
        if not ignore_costs:
            oracle.get_annotation(path, weight=0.4, num_anno=simulation_repetitions)

        # exectue simualtion (multiple times)
        if simulated_label is None:
            simulated_label = np.zeros((len(soft_gt)))
            original_proposed_class = proposed_class
            for j in range(simulation_repetitions):

                # workaorund if distribution is given instead of
                proposed_class = original_proposed_class if soft_proposed_class is None else np.random.choice(
                    [i for i in range(len(soft_proposed_class))],
                    p=soft_proposed_class)

                # calculate proposal acceptance
                accept_rate = proposal_acceptance_offset + ((0.99 - proposal_acceptance_offset)) * soft_gt[proposed_class]

                simulated_class = -1
                # idea: accept rate increases with raising soft gt value
                if oracle.r.random() <= accept_rate:
                    simulated_class = proposed_class
                    # print(simulated_class, soft_proposed_class)
                else:

                    # idea 2: select based on soft gt
                    max_value = 1 - soft_gt[proposed_class]  # proposed can not be selected anymore
                    rand_select = oracle.r.random() * max_value

                    sum_gt = 0
                    # increase collective probability until rand_select is smaller
                    for k, g in enumerate(soft_gt):
                        if proposed_class != k:
                            # ignore proposed element
                            sum_gt += g  # update collective probability
                            if rand_select <= sum_gt:
                                simulated_class = k
                                break

                    assert simulated_class != -1

                simulated_label[simulated_class] += 1


        if simulation_repetitions == 0:
            return simulated_label

        if correct_with_offset < 0:

            # No corrections
            return simulated_label / simulation_repetitions
        else:
            # correct with estimated offset

            # need to lower this score based on confirmation offset
            pc = simulated_label[proposed_class] / simulation_repetitions
            corrected = (pc - correct_with_offset) / (0.99 - correct_with_offset)
            corrected = min(1, max(0, corrected))

            m = max(1, simulation_repetitions - simulated_label[proposed_class])
            p = simulated_label / m

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
        load the specified datasets and enable simulated or real proposal generation
       :return:
    """

    # iterate over specified folders
    folders = FLAGS.folders

    # wildcard check
    if len(folders) == 1 and "*" in folders[0]:
        # print(os.listdir(FLAGS.output_folder))
        paths = glob.glob(join("/data/output_datasets", folders[0]))
        folders = sorted([path.split("/")[-1] for path in paths])
        print(f"Found the folders {folders} with pattern search")


    # log results
    report = None

    # for folder, org_dataset in zip(folders, orig_datasets):
    for folder in folders:

        try:
            alg = SimulatedProposalAcceptance(folder, report=report)
            alg.apply_algorithm()
            report = alg.report
        except Exception as e:
            print(e)
            print("ERROR: did not complete!")


    if report is not None:
        report.show()


if __name__ == '__main__':
    app.run(main)