import os
from os.path import join

from absl import app
from absl import flags

from src.evaluation.evaluate import evaluation_function
from src.util.mixed import get_all_dataset_files
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
import ray
from ray.tune.suggest import ConcurrencyLimiter
FLAGS = flags.FLAGS


flags.DEFINE_list(name='datasets',
                  help='the datasets which should be evaluated', default=[])


def main(argv):
    print("Tune for DCIC Benchmark")

    mode = 'soft'

    datasets = FLAGS.datasets

    ray.init(local_mode=True)

    b_configs = {}

    for dataset_name in datasets:

        # tune on fold 1 on no_alg-10-1.00 subset
        folder = f"{dataset_name}-no_alg-10-1.00"

        for weights in [None,'imagenet']:

            print(f"Tune on dataset {dataset_name} with {weights}")

            input_dataset_files = get_all_dataset_files(FLAGS.output_folder, folder)
            num_slices = len(input_dataset_files)

            assert num_slices <= 5, "Expects only up to 5 splits but received %d" % num_slices

            # iterate over all inputs
            for v_fold_index, f in enumerate(input_dataset_files):

                if v_fold_index != 0:
                    continue # tune only on first fold


                # do hyper parameter optimization
                algo = HyperOptSearch(
                    points_to_evaluate=[{
                            # not pretrained
                            'weights': weights,
                            'batch_size': 128,
                            'epochs': 20,
                            'lr': 1e-1,
                            'dropout': 0.5,
                            'network':  'resnet50v2',
                            'augmentation': 0,
                            'opt': 'sgd',
                            'input_upsampling': True,
                            'weight_decay': 5e-4,
                            'use_class_weights': True,'num_slices': num_slices,'folder': folder,
                            'mode': mode, 'file': f,  'tuning': True, 'v_fold_index':0,
                            'verbose': FLAGS.verbose, 'wandb_usage': FLAGS.wandb,
                            'provided_test': FLAGS.provided_test, 'output_folder': FLAGS.output_folder
                        },
                        {
                                # pretrained
                            'weights': weights,
                            'batch_size': 128,
                            'epochs': 20,
                            'lr': 1e-5,
                            'dropout': 0.5,
                            'network': 'resnet50v2',
                            'augmentation': 0,
                            'opt': 'adam',
                            'input_upsampling': True,
                            'weight_decay': 5e-4,
                            'use_class_weights': True, 'num_slices': num_slices, 'folder': folder,
                            'mode': mode, 'file': f, 'tuning': True, 'v_fold_index': 0,
                            'verbose': FLAGS.verbose, 'wandb_usage': FLAGS.wandb,
                            'provided_test': FLAGS.provided_test, 'output_folder': FLAGS.output_folder
                        },
                    ]
                )
                algo = ConcurrencyLimiter(algo, max_concurrent=1)
                analysis = tune.run(
                    evaluation_function,
                    metric="kl",
                    mode="min",
                    search_alg=algo,
                    num_samples=100,
                    resources_per_trial={'gpu': 1, 'cpu':16,},
                    config={
                        'weights': weights,
                        'batch_size': tune.choice([128]),
                        'epochs': tune.choice([20]),
                        'lr': tune.choice([1e-1,3e-1,1e-4,1e-5]),
                        'dropout': tune.choice([0, 0.5]),
                        'network': tune.choice(['wideresnet28-10', 'resnet50v2_large', 'densenet121', 'incepresv2', 'resnet50v2', 'wideresnet16-8']),
                        'augmentation': tune.choice([0,1]),
                        'opt': tune.choice(['sgd', 'adam', 'sgdw', 'sgdwr']),
                        'input_upsampling': tune.choice([True, False]),
                        'weight_decay': tune.choice([5e-4, 1e-3]),
                        'use_class_weights': True,'num_slices': num_slices,'folder': folder,
                        'mode': mode, 'file': f,  'tuning': True, 'v_fold_index':0,
                        'verbose': FLAGS.verbose, 'wandb_usage': FLAGS.wandb,
                        'provided_test': FLAGS.provided_test, 'output_folder': FLAGS.output_folder,
                        'slices':FLAGS.slices
                    },
                    raise_on_failed_trial=False
                )

                print("Best config: ", analysis.best_config)
                b_configs[f"{dataset_name}-{weights}"] = analysis.best_config

                # Get a dataframe for analyzing trial results.
                df = analysis.results_df
                # print(df)


                path = "./tuning_logs/"
                os.makedirs(path,exist_ok=True)
                df.to_csv(join(path,f"{dataset_name}_{weights}.csv"))


    print(b_configs)

if __name__ == '__main__':
    app.run(main)
