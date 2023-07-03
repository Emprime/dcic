# Benchmark

This is the official repo of the Data-Centric Image Classification (DCIC) Benchmark.
The goal of this benchmark is to measure the impact of tuning the dataset instead of the model for a variety of image classification datasets.

![datasets overview](images/datasets.png)

## Citation

Please cite as
```
@article{schmarje2022benchmark,
    author = {Schmarje, Lars and Grossmann, Vasco and Zelenka, Claudius and Dippel, Sabine and Kiko, Rainer and Oszust, Mariusz and Pastell, Matti and Stracke, Jenny and Valros, Anna and Volkmann, Nina and Koch, Reinahrd},
    journal = {36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks},
    title = {{Is one annotation enough? A data-centric image classification benchmark for noisy and ambiguous label estimation}},
    year = {2022}
}
```

Please see the full details about the used datasets below, which should also be cited as part of the license.

## Installation

All provided code should be run with docker and thus a working docker version with GPU support is expected. 
The source code is tested on a RTX 3090 with 24GB VRAM and might not work with other hardware especially with lower VRAM.

First, create an image with docker file

    docker build -t dcic .
    
Hint: If the command fails with a GPG error, open Dockerfile and follow the instructions in line 3.

All following docker commands should be executed within the folder of this downloaded project.
   
You have to download the raw data into a location of your choice `<DATASET_ROOT>`. Please see the information about datasets at the end.


After this you set the correct values for your system in `./run.sh`
This includes own [Weights & Bias](https://wandb.ai/) API Token if you want to log to their system.
 
Before you can use any other commands you need to check and setup the benchmark by calling 
```
    ./run.sh src.util.check_and_setup
```

If you run the code WITHOUT the wrapper script `run.sh` and docker you need to change the root_directory with the command line argument --data_root
You can also change the default locations of the input and evaluation datasets by specifing the corresponding command line argument
See the file `dataset_skeleton.py` for more information.

If you encounter any issues or errors, get more information from the error log on the command line.
The desired file structure after this setup is defined below.

### Folds & Slices & Splits

For every dataset the benchmark provides 5 folds which contain roughly 20% of the complete data. 
These 5 folds allows a 5-fold crossvalidation. Fold 1 describes a certain fifth of the data and the slice 1 describes the dataset where fold 3-5 for trainng, fold 1 for  validation and fold 2 for test is used.
Split is the separation into training, validation, unlabeled and test.

The general pipeline is described below for one slice but can be adopted to any split by changing the <v_fold>.
In general only the v_fold 1,2,3 are evaluated to save runtime.

### Expected File structure

```
<DATASET_ROOT> # aka /data/ inside docker 
 | 
 ---- raw_datasets/  # A subfolder for every image included 
 |        |
 |        |--- MiceBone/
 |        |    |
 |        |    |--- fold1/ # All images inside folder
 |        |    |--- fold2/
 |        |    |--- fold3/
 |        |    |--- fold4/
 |        |    |--- fold5/ 
 |        |    |--- annotations.json # Raw Annotations
 |        |--- Plankton/
 |        |---- ...
 |
 ---- input_datasets/ # input for the algorithms
 |       |
 |       | --- MiceBone/ # 3 files for the 3 split-datasets (see below for the defintion of split)
 |       | --- Plankton/
 |       | ---- ...
 ---- evaluate_datasets/ # evaluation datasets for the final evaluation (DO NOT use during training)
 |        |
 |        |--- MiceBone/ # 3 files for the 3 split-datasets (similar to the input dataset but with perfect gt)
 |        |--- Plankton/
 |        |---- ...
 ----- output_datasets/ # outputs of all implemented algorithms
          | 
          | --- <FOLDER1> # arbitary folder name with 3 datasets based on the 3 split datasets
          | --- <FOLDER2>
          | --- ....  
            

```


## Usage 

The benchmark is structured into two phases: Labeling and Evaluation

![](images/outline.png)

### 1. Labeling

In this step you can apply your own algorithm to the sliced dataset to generate a newly labeled dataset for the evaluation.

We generally work in two steps:
- initial annotations
- refine annotations

Both phases are structured an organized in the common `AlgorithmSkeleton` class.
We provide a default implementation for the intitial annotations in the method `get_initial_annotations()` and provide a dummy method `run()` for the refining of the annotations. 
For more informations about implementing your own method see below.

The algorithm is applied to all specified datasets with `dataset` (default: `all` datasets used) and slices with valdidation_vald `v_fold` (default: -1, uses all slices).
Attention, you can specify less slices than 3 but the evaluation expects no gaps between the slice numbers. For example slices [1,2,4] would be evaluated wrongly.
The outputs are stored in a folder based on the `name` of the algorithm class. The output consists of all processed dataset slices which are based on the class `DatasetDCICJson`.
For more details about the parameters see `algorithm_skeleton.py`.

The initial annotations are created by using `number_annotations_per_image` annotations for a random image (in training and validation) until the `percentage_labeled_data` is reached. 
The arguments can be addressed via the command line.
The annotations are taken from an `AnnotationOracle` which is described in `src/util/oracle.py`.

The refine annotations step is custom to all methods. 
We provide several baselines as example and comparison which are all described in the main main paper.

Notes:
- `no_algorithm.py` use only initial annotations and thus can be used as a baseline
- `pseudo_label.py` trains a ResNet50v2 on the provided labeled data and predict the labels of all other labels. It is called Pseudo v2 in the paper and its variants can be addressed via command line arguments
- MeanTeacher, Pi Model, Fixmatch and Pseudolabel v1 (+ S2C2) were executed based on the original authors source code (TODO: Insert currently unpublished Link, Under Review). You can import the generated predictions as described below.


```
# In general the algorithm is called with 
./run.sh src.algorithms.<FILE>


# Generating Baseline Results can be achieved for example with
./run.sh src.algorithms.no_algorithm

# You can add additional arguments to influence the intitial annotation and datasetsn
# these arguments are here explained nor no_algorithm but work for all algorithms

# These genereates all datasets, labeled percentage and number annotations possiblities
./run.sh src.algorithms.no_algorithm --datasets all --percentage_labeled_data -1 --number_annotations_per_image -1

# You can specify multiple datasets or specific labeled_percentage or annotations like this 
./run.sh src.algorithms.no_algorithm --datasets Turkey,MiceBone,QualityMRI --percentage_labeled_data 1.0 --number_annotations_per_image 1

# Many other algorithm can be started similarly as shown below
# Pseudolabel (Predict labels of unlabeled data)
./run.sh src.algorithms.pseudo_label
# DivideMix (Mixture of Semi-Supervised Learning and Noise Estimation)
./run.sh src.algorithms.divide_mix

# If you use external scripts and need to import the predictions a loader is defined in `predictions_import`. See the file for more information
# Attention: you need to paste the predictions under <DATASET_ROOT>/manual_predictions
./run.sh src.algorithms.predictions_import --root_directory_predictions /data/manual_predictions --subset train#ema_normal --subset val#ema_normal --subset unlabeled#ema_normal --subset test#ema_normal --number_annotations_per_image -1 --percentage_labeled_data -1


```

### 2. Evaluate

The evaluation is calculated on a predefined model and hyperparameters with the given input data.

Run the evaluation inside docker with the function `./run.sh src.evaluation.evaluate`. This scripts has the following parameters:
- `folders` are the folders under `output_folder` which should be evaluated
- `output_folder` the path where the created `DatasetDCICJSON` for all specified slices and datasets are stored (default: /data/output_datasets)
- `mode` the type of input labels can be hard or soft. The loss is `Cross-entropy`for hard labels and `Kullback Leiber Divergence`for soft labels. (default: ['soft']) 
- `verbose` adjust the verbosity for the evaluation, 0 only final result table, 1 results per experiment (loaded folder), 2 results per slice, 3 results per slice + training logs

The reported metrics are `kappa`, `kl`, `f1`, `acc` for the given (weighted) budget.
- `kappa` consistency score between slices which are used as input. For this calculation the class `unknown` can be ignored or considered. The default behaviour is that in the final metric the class `unknown` is considered.
- `kl` Kullback leiber divergence between soft gt and predicted output to measure the quality of soft predictions (interesting for soft labels). 
- `f1` Macro F1-Score over all classes as a classification performance (interesting for hard labels). 
- `acc` Is the accuracy for each class averaged over the classes to compensate for class imbalance

The classification scores like `kl,f1,acc` can be calculated on the original or provided (possibly relabeled) test data. The default behaviour is that in the final metric the original test data is used.
Furthermore additional outputs like the consistency matrices and a classification report are provided.


Example:
```
# Run evaluation on one folder in `output_datasets`
./run.sh src.evaluation.evaluate --folders <NAME> --mode soft

# ... multiple folders
./run.sh src.evaluation.evaluate --folders <NAME>,<NAME2> --mode soft


# ... with wild cards (if you provide exactly one folder)
./run.sh src.evaluation.evaluate --folders *pseudo*


# you can add additional logging to Weights & Biases
./run.sh src.evaluation.evaluate --folders *pseudo* --wandb 

    
```

## Development

### Debug Visualization

You can visualize a dataset json file by using the visualizer like so

```
# you may use wild cards
./run.sh src.util.visualize --file /data/output_datasets/*divide_mix-01-1.00/*
```

### Tuning Hyperparameters for evaluation

The hyperparameters for the evaluation phase were determined by
 
```
./run.sh src.evaluation.tune --datasets <DATASET_NAME>
```

The best set of parameters was determined and inserted into `src/datasets/dataset_skeleton.py` under `hyper_parameters`.

## Contributing

### Implement your own method

If you want to implement you have to interherit from `AlgorithmSkeleton`. You can orientate yourself on the minimal working solution below.
You can other methods from `AlgorithmSkeleton` like `get_initial_annotations` or `before_all_slices_processed` to change the initial annotation scheme or for callbacks before or after certain events.

```
from absl import app
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton


class Algorithm(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self,'no_alg')

    def run(self, raw_ds, new_ds, oracle, dataset_info, v_fold):
        # put your changes here
        # you can get annotations with oracle.get_annotation(file_name,weight)
        # you can update the label with new_ds.update_image(path, split, soft_gt)
        
        return new_ds

def main(argv):
    """
       Apply only initial annotation
       :return:
    """

    alg = Algorithm()
    alg.apply_algorithm()



if __name__ == '__main__':
    app.run(main)
```

If you have all file names in a list `paths` and all new soft_gt values in a list `preds` you can update the dataset with the following loop inside the method run
```
    for i, path in enumerate(paths):
        split = ds.get(path,'original_split')  # determine original split before move to unlabeled
        ds.update_image(path, split, [float(temp) for temp in preds[i]])
```

### Add a new Dataset

If you want to add a new Dataset you have to implement a class which inherits from DatasetSkeleton. 
You also have to register the new dataset in `src/utils/const.py` under `get_all_dataset_infos()`.

The appropriate hyperparameters during the evaluation phase have to be defined under `hyper_parameters` in `src/datasets/dataset_skeleton.py`.

## Datasets and their citation

The datasets are available at [https://doi.org/10.5281/zenodo.7152309](https://doi.org/10.5281/zenodo.7152309)

As part of the license the original works and authors of the datasets should be acknowledgement. Please add the following references to be compliant with the license:

```

@article{schoening2020Megafauna,
author = {Schoening, T and Purser, A and Langenk{\"{a}}mper, D and Suck, I and Taylor, J and Cuvelier, D and Lins, L and Simon-Lled{\'{o}}, E and Marcon, Y and Jones, D O B and Nattkemper, T and K{\"{o}}ser, K and Zurowietz, M and Greinert, J and Gomes-Pereira, J},
doi = {10.5194/bg-17-3115-2020},
journal = {Biogeosciences},
number = {12},
pages = {3115--3133},
title = {{Megafauna community assessment of polymetallic-nodule fields with cameras: platform and methodology comparison}},
volume = {17},
year = {2020}
}

@article{Langenkamper2020GearStudy,
author = {Langenk{\"{a}}mper, Daniel and van Kevelaer, Robin and Purser, Autun and Nattkemper, Tim W},
doi = {10.3389/fmars.2020.00506},
issn = {2296-7745},
journal = {Frontiers in Marine Science},
title = {{Gear-Induced Concept Drift in Marine Images and Its Effect on Deep Learning Classification}},
volume = {7},
year = {2020}
}


@article{peterson2019cifar10h,
author = {Peterson, Joshua and Battleday, Ruairidh and Griffiths, Thomas and Russakovsky, Olga},
doi = {10.1109/ICCV.2019.00971},
issn = {15505499},
journal = {Proceedings of the IEEE International Conference on Computer Vision},
pages = {9616--9625},
title = {{Human uncertainty makes classification more robust}},
volume = {2019-Octob},
year = {2019}
}

@article{schmarje2019,
author = {Schmarje, Lars and Zelenka, Claudius and Geisen, Ulf and Gl{\"{u}}er, Claus-C. and Koch, Reinhard},
doi = {10.1007/978-3-030-33676-9_26},
issn = {23318422},
journal = {DAGM German Conference of Pattern Regocnition},
number = {November},
pages = {374--386},
publisher = {Springer},
title = {{2D and 3D Segmentation of uncertain local collagen fiber orientations in SHG microscopy}},
volume = {11824 LNCS},
year = {2019}
}

@article{schmarje2021foc,
author = {Schmarje, Lars and Br{\"{u}}nger, Johannes and Santarossa, Monty and Schr{\"{o}}der, Simon-Martin and Kiko, Rainer and Koch, Reinhard},
doi = {10.3390/s21196661},
issn = {1424-8220},
journal = {Sensors},
number = {19},
pages = {6661},
title = {{Fuzzy Overclustering: Semi-Supervised Classification of Fuzzy Labels with Overclustering and Inverse Cross-Entropy}},
volume = {21},
year = {2021}
}

@article{schmarje2022dc3,
author = {Schmarje, Lars and Santarossa, Monty and Schr{\"{o}}der, Simon-Martin and Zelenka, Claudius and Kiko, Rainer and Stracke, Jenny and Volkmann, Nina and Koch, Reinhard},
journal = {Proceedings of the European Conference on Computer Vision (ECCV)},
title = {{A data-centric approach for improving ambiguous labels with combined semi-supervised classification and clustering}},
year = {2022}
}


@article{obuchowicz2020qualityMRI,
author = {Obuchowicz, Rafal and Oszust, Mariusz and Piorkowski, Adam},
doi = {10.1186/s12880-020-00505-z},
issn = {1471-2342},
journal = {BMC Medical Imaging},
number = {1},
pages = {109},
title = {{Interobserver variability in quality assessment of magnetic resonance images}},
volume = {20},
year = {2020}
}


@article{stepien2021cnnQuality,
author = {St{\c{e}}pie{\'{n}}, Igor and Obuchowicz, Rafa{\l} and Pi{\'{o}}rkowski, Adam and Oszust, Mariusz},
doi = {10.3390/s21041043},
issn = {1424-8220},
journal = {Sensors},
number = {4},
title = {{Fusion of Deep Convolutional Neural Networks for No-Reference Magnetic Resonance Image Quality Assessment}},
volume = {21},
year = {2021}
}

@article{volkmann2021turkeys,
author = {Volkmann, Nina and Br{\"{u}}nger, Johannes and Stracke, Jenny and Zelenka, Claudius and Koch, Reinhard and Kemper, Nicole and Spindler, Birgit},
doi = {10.3390/ani11092655},
journal = {Animals 2021},
pages = {1--13},
title = {{Learn to train: Improving training data for a neural network to detect pecking injuries in turkeys}},
volume = {11},
year = {2021}
}

@article{volkmann2022keypoint,
author = {Volkmann, Nina and Zelenka, Claudius and Devaraju, Archana Malavalli and Br{\"{u}}nger, Johannes and Stracke, Jenny and Spindler, Birgit and Kemper, Nicole and Koch, Reinhard},
doi = {10.3390/s22145188},
issn = {1424-8220},
journal = {Sensors},
number = {14},
pages = {5188},
title = {{Keypoint Detection for Injury Identification during Turkey Husbandry Using Neural Networks}},
volume = {22},
year = {2022}
}



```
## TODO
- make datasets automatically downloadable 

    



