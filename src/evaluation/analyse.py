import matplotlib
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from absl import flags
from matplotlib.lines import Line2D
from scipy.stats import sem
from sklearn.linear_model import LinearRegression

from src.evaluation.report import DCICReport
import scipy
import warnings
np.set_printoptions(suppress=True)
"""
Analyse of Benchmark data


We analyze by selecting and grouping the datasets and methods for different selections of annotation schemes.
We generate the results with the following algorithm.
1. We select the results for every dataset and method combination with the defined annotation schemes.
2. Per dataset and method we order the results based on the annotations and report incremental improvements between their averaged scores over the folds
3. We average and count the values across the grouped methods and then datasets.
"""

warnings.filterwarnings('ignore')


all_datasets = "Benthic,CIFAR10H,MiceBone,Pig,Plankton,QualityMRI,Synthetic,Treeversity#1,Treeversity#6,Turkey"
grouped_datasets = "Benthic.CIFAR10H.MiceBone.Pig.Plankton.QualityMRI.Synthetic.Treeversity#1.Treeversity#6.Turkey"
easy_datasets = "CIFAR10H.Plankton.Synthetic.Treeversity#1.Turkey"
diff_datasets = "Benthic.MiceBone.Pig.QualityMRI.Treeversity#6"

multi_group = ",".join([grouped_datasets, easy_datasets, diff_datasets])

ssl_methods="pseudo.mean.mean_s2c2.pi.pi_s2c2.pseudo_ssl.pseudo_s2c2.fixmatch.fixmatch_s2c2.divide_mix"
self_methods="simclr.moco.byol.swav" # simclr_old
noise_methods="elr.sgnp.het"
soft_methods="no_alg.pseudo_soft.pseudo_soft_no_pretrain"
hard_methods=ssl_methods + "." + self_methods + "."+ noise_methods
old_methods="elr.het.sgnp.divide_mix.fixmatch.fixmatch_s2c2.mean.mean_s2c2.pi.pi_s2c2.pseudo_ssl.pseudo_s2c2.pseudo.pseudo_soft_no_pretrain.byol.moco.simclr.swav"
grouped_methods = "no_alg.elr.het.sgnp.divide_mix.fixmatch.fixmatch_s2c2.mean.mean_s2c2.pi.pi_s2c2.pseudo_ssl.pseudo_s2c2.pseudo.pseudo_soft.pseudo_soft_no_pretrain.byol.moco.simclr.swav."



all_methods = ",".join(grouped_methods.split("."))




colors = ['darkslategray',   'red',  'gold',
                      'lawngreen', 'g',  'turquoise',  'b', 'blueviolet',
                      'pink', 'm', 'coral', 'lime', 'sandybrown','darkkhaki', 'dodgerblue','saddlebrown','skyblue','orchid', ]
_markers = ['x','*', '+',  'o','<','>','v','^' , "1","2", "3","4" ]
line_styles = ['-','--','-.',':']

logs_folder = "/data/evaluation_logs"


def analyze(report, exp_name, dataset, method, annos, score_names, value_normalization='absolute',  x_axis='budget', y_axis='metric', markers=['<','>','v','^' ], no_line = False, mean_stds=False,short=True, correlation=False,custom_scale=None):
    """
    The analyze function is responsible for grouping, analyzing, and visualizing data based on specified parameters.
    It takes in a DCICReport instance, experiment name, dataset groups and selections, method groups and selections,
     annotation selections, score names, and various optional parameters.

The function performs the following tasks:

    1. Converts the dataset, method, and annotation selections into grouping and selection lists.
    2. Collects the desired information from the DCICReport instance based on the selections and groupings.
    3. Calculates relative improvements and standard deviations for each score name.
    4. Normalizes the values based on the specified normalization type.
    5. Stores the results in a dictionary.
    6. Calls the create_plots function to generate visualizations based on the collected results.
    7. If specified, calculates and displays correlations between the scores.
    8. Saves the generated plots.

    :param report: An instance of the DCICReport class containing the data to be analyzed.
    :param exp_name: The name of the experiment or visualization.
    :param dataset: A string specifying the dataset groups and selections. Comma-separated values represent different groups, and dots represent groupings within a group.
    :param method: A string specifying the method groups and selections. Comma-separated values represent different groups, and dots represent groupings within a group.
    :param annos: A string specifying the annotation selections.
    :param score_names: A list of score names to be analyzed and visualized.
    :param value_normalization: Specifies the type of value normalization. Options are 'absolute' (absolute values), 'relative_baseline' (relative to baseline), and 'relative_anno' (relative to the first annotation).
    :param x_axis: Specifies the data to be plotted on the x-axis. Default is 'budget'.
    :param y_axis: Specifies the data to be plotted on the y-axis. Default is 'metric'.
    :param markers: A list of markers to be used between different configurations on the y-axis.
    :param no_line: Boolean flag indicating whether to plot lines. Default is False.
    :param mean_stds: Boolean flag indicating whether to use means of standard deviations or standard deviations of means.
    :param short: Boolean flag indicating whether to use short text instead of long text with debug information
    :param correlation: Boolean flag indicating whether to calculate and display correlations between scores.
    :param custom_scale: Custom scaling function to be applied to the data.
    :return:
    """

    sort_by = 'none' # do not use, too much work for the benefit


    missing_value = np.nan

    # convert strings to groupings and selections
    datasets = [[k for k in t.split(".")] for t in dataset.split(",")]
    methods = [[k.replace("$",".") for k in t.split(".")] for t in method.split(",")]
    annos = [t for t in annos.split(",")]

    print("\n", exp_name, ": Selection and Groupings ", datasets, methods, "\n")

    # collect the desired infos
    results = {}
    for dataset_group in datasets:

        for d in dataset_group:


            for method_group in methods:
                group_results = []
                for m in method_group:

                    if f"{d}-{m}" in results.keys():
                        continue

                    # relative improvements and stds per score name
                    temp = {score_name : [[] for i in range(len(annos)*3)] for score_name in score_names}
                    start_value = {score_name: missing_value for score_name in score_names}

                    for i,a in enumerate(annos):
                        # define desired experiment name and check if available
                        exp = f"{d}-{m}-{a}"


                        # special case spa in name
                        if "spa%03d" in m:
                            # insert number of supervision in spa term
                            num_annos = int(a.split("-")[0])
                            temp_method = m % num_annos
                            temp_a = "01-"+ a.split("-")[1]
                            exp = f"{d}-{temp_method}-{temp_a}"


                        cs = report.scores.get(exp,{})
                        baseline_cs = report.scores.get( f"{d}-no_alg-{a}", {})

                        # print(report.scores.keys(), exp)

                        for score_name in score_names:

                            temp[score_name][3 * i] = missing_value
                            temp[score_name][3 * i + 1] = missing_value

                            if len(cs) > 0:
                                values = report._calculate_values_from_scores(score_name, cs.get(score_name,[]), verbose=0)

                                if len(values) > 0:

                                    # determine start value
                                    if value_normalization == 'relative_anno' and np.isnan(start_value[score_name]):
                                        start_value[score_name] = np.mean(values)

                                    if value_normalization == 'absolute':
                                        start_value[score_name] = 0

                                    if value_normalization == 'relative_baseline':
                                        start_value[score_name] = np.mean(report._calculate_values_from_scores(
                                            score_name, baseline_cs.get(score_name,[]), verbose=0))

                                    if not np.isnan(start_value[score_name]):
                                        values -= start_value[score_name]

                                    temp[score_name][3 * i] = np.mean(values)
                                    temp[score_name][3 * i + 1] = np.std(values)
                                    temp[score_name][3 * i + 2] = values




                    results[f"{d}-{m}"] = temp

    # plot different versions of the lines
    create_plots(annos, score_names, datasets, methods, results,x_axis,y_axis,sort_by,
                 mean_stds,no_line, exp_name,short, markers, custom_scale,mode="median")
    create_plots(annos, score_names, datasets, methods, results,x_axis,y_axis,sort_by,
                 mean_stds,no_line, exp_name, short, markers, custom_scale, mode="mean")

    if correlation:
        print("Correlations:")
        # calculate correlation between scores
        for i in range(len(score_names)):
            x = np.array([
                results[c][score_names[i]][3*k]
                for c in results.keys()
                for k in range(len(annos))
                ])

            for j in range(len(score_names)):
                if i < j:
                    y = np.array([
                            results[c][score_names[j]][3 * k]
                            for c in results.keys()
                            for k in range(len(annos))
                        ])

                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['text.latex.preamble'] = [r'\boldmath']


                    # remove nans
                    is_not_nan = ~np.bitwise_or(np.isnan(x), np.isnan(y))
                    print(f"{score_names[i]} vs. {score_names[j]}, {scipy.stats.pearsonr(x[is_not_nan],y[is_not_nan])}")

                    plt.scatter(y[is_not_nan],x[is_not_nan], marker='x', c='k')#_markers[is_not_nan],_colors[is_not_nan]

                    # linear regression
                    X = x[is_not_nan].reshape((-1, 1))
                    Y = y[is_not_nan].reshape((-1, 1))
                    reg = LinearRegression().fit(Y, X)
                    pred = reg.predict(Y)
                    mae = np.abs((pred - X))
                    # print(mse.shape, mse[:10])
                    print(f"Linear regression ({score_names[j]} -> {score_names[i]}): R^2 {reg.score(Y,X):0.04f} MAE {np.mean(mae):0.04f} +- {np.std(mae):0.04f} [{np.min(mae):0.04f}-{np.max(mae):0.04f}], coef {reg.coef_}, intercept {reg.intercept_}")

                    x_line = np.array([np.min(Y),np.max(Y)])
                    y_line = np.array([reg.intercept_[0] + reg.coef_[0][0] * np.min(Y), reg.intercept_[0] + reg.coef_[0][0] * np.max(Y)])
                    plt.plot(
                        x_line,
                        y_line,
                        c="r", marker="", linestyle="-")


                    # print( np.std(mae))
                    plt.fill_between(x_line, y_line - np.mean(mae), y_line + np.mean(mae), color='r', alpha=0.2)

                    plt.xlabel(get_show_label_name(score_names[j]))
                    plt.ylabel(get_show_label_name(score_names[i]))


                    # save plot
                    plt.savefig(join(logs_folder, "images", f"{exp_name}_{score_names[j]}_{score_names[i]}.png"),bbox_inches='tight',pad_inches = 0)
                    plt.clf()

                    plt.rcParams['text.usetex'] = False
    print("\n\n")


def create_plots(annos, score_names, datasets, methods, results, x_axis, y_axis,
                 mean_stds, no_line, exp_name,short, markers,custom_scale, mode='median'):

    """
    The create_plots function provides a way to visualize and compare results for different score names, datasets, and methods. It generates plots that can help in understanding trends and patterns in the data, allowing for easier analysis and interpretation. It also creates a table with the call of 'add_row_line*
     :param annos: A list of annotation levels or budgets.
    :type annos: list[str]

    :param score_names: A list of score names to plot.
    :type score_names: list[str]

    :param datasets: A list of dataset groups, where each group is a list of dataset names. This defines which elements should be aggregated
    :type datasets: list[list[str]]

    :param methods: A list of method groups, where each group is a list of method names.This defines which elements should be aggregated
    :type methods: list[list[str]]

    :param results: A dictionary containing the results data for different dataset-method combinations.
    :type results: dict

    :param x_axis: The parameter to use for the x-axis. Possible values are 'annos', 'dataset', 'method', or 'budget'.
    :type x_axis: str

    :param y_axis: The parameter to use for the y-axis. Possible values are 'metric', 'dataset', or 'method'.
    :type y_axis: str

    :param mean_stds: Whether to show mean and standard deviation values. True to show, False to hide.
    :type mean_stds: bool

    :param no_line: Whether to show lines connecting the data points. True to hide lines, False to show.
    :type no_line: bool

    :param exp_name: The name of the experiment or plot.
    :type exp_name: str

    :param short: Whether to use short labels for dataset and method names. True to use short labels, False to use full names.
    :type short: bool

    :param markers: A list of markers to use for different dataset or method groups.
    :type markers: list[str]

    :param custom_scale: The custom scale to use for the y-axis. None for default scale.
    :type custom_scale: str or None

    :param mode: The mode of the plot. Possible values are 'median' or any other mode specified in the data.
    :type mode: str, optional

    :return: None
    """

    seperator = "&"
    end = "\\\\"
    first_padding = 20
    padding = 20
    verbose = mode == 'median'

    if custom_scale is not None:
        from matplotlib.ticker import NullFormatter, FixedLocator
        fig, axs = plt.subplots()
        axs.set_yscale('function', functions=custom_scale)
        axs.yaxis.set_major_locator(FixedLocator(np.arange(-0.4, .1, 0.05)))

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\boldmath']

    # determine x axis
    if x_axis == 'annos':
        xs = annos
    elif x_axis == 'dataset':
        for dg in datasets:
            assert len(dg) == 1, "all datsets should be viewed individually if specified as x-axis"
        xs = [get_show_label_name(dg[0]) for dg in datasets]
        assert  len(annos) == 1
    elif x_axis == 'method':
        for m in methods:
            assert len(m) == 1, "all methods should be viewed individually if specified as x-axis"
        xs = [get_show_label_name(m[0]) for m in methods]
        assert len(annos) == 1
    elif x_axis == 'budget':
        xs = [f'{int(int(a.split("-")[0]) * float(a.split("-")[1]) * 100)}\%' for a in annos]
    else:
        xs = None

    # determine y axis and if image is possible
    image_possible = True
    if y_axis == 'metric':
        # assert len(datasets) == 1
        # assert len(methods) == 1
        pass
    elif y_axis == 'dataset':
        if not len(methods) == 1 and x_axis != 'method':
            image_possible = False
            if verbose:
                print("Not possible to create image with 3 axis, methods are too long")
    elif y_axis == 'method':
        if not len(datasets) == 1 and x_axis != 'dataset':
            image_possible = False
            if verbose:
                print("Not possible to create image with 3 axis, datasets are too long")
    else:
        xs = None

    # print averaged results
    if verbose:
        print(
            f"{''.ljust(first_padding)} {seperator} {f' {seperator} '.join([x.ljust(padding) for x in xs])} {end}"
        )


    score_colors = {'kl': 'tab:blue', 'ece': 'tab:red', 'macro_acc': 'tab:green',
                       'macro_f1': 'tab:purple', 'input_consistency': 'tab:orange',
                       'input_kl': 'skyblue', 'input_ece': 'lightcoral',
                       'input_macro_acc': 'lawngreen', 'input_macro_f1': 'violet', }


    for c_index, score_name in enumerate(score_names):
        # show results

        score_color = score_colors[score_name]

        if  y_axis != 'metric' and verbose:
            # give header if not row name
            print(f"\midrule{end}Results for ", score_name, end)

        for m_index, dataset_group in enumerate(datasets):

            dataset_group_name = ".".join([d for d in dataset_group])

            if y_axis != 'dataset' and len(datasets) > 1 and len(methods) > 1 and x_axis != 'method' and verbose:
                print("Dataset ", get_show_label_name(dataset_group_name))


            for l_index, method_group in enumerate(methods):



                # calculate applicable dataset and method combinations
                selected_combis = [f"{d}-{m}" for d in dataset_group for m in method_group]

                method_group_name = ".".join([m for m in method_group])

                if y_axis != 'method' and len(datasets) > 1  and len(methods) > 1 and x_axis != 'dataset' and verbose:
                    print("Method ", get_show_label_name(method_group_name))


                if y_axis == 'metric':
                    row_name = score_name
                elif y_axis == 'dataset':
                    row_name = dataset_group_name
                elif y_axis == 'method':
                    row_name = method_group_name
                else:
                    row_name = "INVALID"

                if x_axis == 'dataset':

                    selected_results = np.array([
                        [results[f"{d[0]}-{m}"][score_name][i] for d in datasets for i in range(3)]
                        for m in method_group
                    ])

                elif x_axis == 'method':

                    # [f"{d}-{m}" for d in dataset_group for m in method_group]
                    selected_results = np.array([
                        [results[f"{d}-{m[0]}"][score_name][i] for m in methods for i in range(3)]
                        for d in dataset_group
                    ])

                else:
                    # last case annos or budget

                    # collect scores based on score name nad cobmis
                    selected_results = np.array([
                        results[c][score_name]
                        for c in selected_combis
                    ])


                # detmine line index

                if y_axis == 'dataset':
                    index = m_index
                elif y_axis == 'method':
                    index = l_index
                else:
                    index = 0

                add_row_line(row_name, xs, selected_results, score_color, index, short, no_line, mode,
                             mean_stds, padding, first_padding, seperator, end, markers, verbose, annos)


                if x_axis == 'method':
                    break
            if x_axis == 'dataset':
                break


    if image_possible:


        plt.xlabel(x_axis if x_axis != 'annos' else "$m-n$")


        # plt.xlabel(x_label)
        file_name = f"{exp_name}.png" if mode == "median" else f"{exp_name}_{mode}.png"
        plt.savefig(join(logs_folder, "images", file_name), bbox_inches='tight', pad_inches=0)
        plt.clf()

        # create legend

        # color is score
        # line type is y_axis
        # markers are given

        legend_elements = []
        for score_name in score_names:
            legend_elements += [Line2D([0], [0], color=score_colors[score_name],
                                   linestyle="", marker='s',
                                   label=get_show_label_name(score_name))]

        # y_axis

        if y_axis == 'dataset':
            y_info = datasets
        elif y_axis == 'method':
            y_info = methods
        else:
            y_info = None

        if y_info is not None:
            for l_index, group in enumerate(y_info):
                name = ".".join([g for g in group])
                l = line_styles[l_index % len(line_styles)]
                m = markers[l_index % len(markers)]
                legend_elements += [Line2D([0], [0], color='k',
                                           linestyle=l, marker=m,
                                           label=get_show_label_name(name))]

        for e in legend_elements:
            if "$" in e.get_label():
                plt.rcParams['text.usetex'] = True
                plt.rcParams['text.latex.preamble'] = [r'\boldmath']


        plt.legend(handles=legend_elements)
        file_name = f"{exp_name}_legend.png"
        plt.savefig(join(logs_folder, "images", file_name), bbox_inches='tight', pad_inches=0)
        plt.clf()

        plt.rcParams['text.usetex'] = False

def get_show_label_name(label):

    # print(label)

    if label == ssl_methods:
        return "Semi-Supervised"
    if label == self_methods:
        return "Self-Supervised"
    if label == noise_methods:
        return "Noise estimation"
    if label == soft_methods:
        return "Using soft labels"
    if label == hard_methods:
        return "Using hard labels"
    if label == diff_datasets:
        return "Difficult datasets"
    if label == easy_datasets:
        return "Easy datasets"

    if label == "mean.pi.fixmatch.pseudo_ssl":
        return "SSL"
    if label == "mean_s2c2.pi_s2c2.fixmatch_s2c2.pseudo_s2c2":
        return "SSL+S2C2"

    if label in grouped_datasets.split("."):
        return label.replace("#",'\#')

    show_names = {'ece': '$ECE$', 'input_ece':'$\hat{ECE}$',
                  'macro_f1': '$F1$','input_macro_f1':'$\hat{F1}$',
                  'macro_acc': '$ACC$', 'input_macro_acc':'$\hat{ACC}$',
                  'input_consistency': '$\kappa$', 'kl': '$KL$',
                  'input_kl': '$\hat{KL}$',
                  'budget': '$b$',
                  'w_budget': '$\hat{b}$',
                  'no_alg':'Baseline', 'simclr':'SimCLR','pseudo':'Pseudo v2 hard','mean':'Mean','mean_s2c2':'Mean+S2C2',
                  'pi':'$\pi$','pi_s2c2':'$\pi$+S2C2', 'pseudo_ssl':'Pseudo v1', 'pseudo_s2c2':'Pseudo v1 + S2C2',
                  'fixmatch':'Fixmatch','fixmatch_s2c2':'Fixmatch + S2C2', 'divide_mix': 'DivideMix',
                  'moco':'MOCOv2','byol':'BYOL','swav':'SWAV','simclr_old':'SimCLR v1','elr':'ELR+',
                  'sgnp':'SGNP','het':'Het','pseudo_soft':'Pseudo v2 soft','pseudo_soft_no_pretrain':'Pseudo v2 not',
                  }


    return show_names.get(label,label)


def add_row_line(name, xs, selected_results, score_color, l_index, short, no_line, mode, mean_stds, padding, first_padding, seperator, end, markers, verbose, annos):
    """
     Adds a row to the plot with the corresponding data points and line.

     :param row_name: The name of the row, which represents the dataset, method, or metric.
     :type row_name: str

     :param xs: The values for the x-axis.
     :type xs: list[str] or None

     :param selected_results: The selected results data for the row.
     :type selected_results: numpy.ndarray

     :param score_color: The color to use for the data points.
     :type score_color: str

     :param index: The index of the row.
     :type index: int

     :param short: Whether to use short labels for dataset and method names. True to use short labels, False to use full names.
     :type short: bool

     :param no_line: Whether to show lines connecting the data points. True to hide lines, False to show.
     :type no_line: bool

     :param mode: The mode of the plot. Possible values are 'median' or any other mode specified in the data.
     :type mode: str

     :param mean_stds: Whether to show mean and standard deviation values. True to show, False to hide.
     :type mean_stds: bool

     :param padding: The padding size for formatting the row.
     :type padding: int

     :param first_padding: The padding size for formatting the first column in the row.
     :type first_padding: int

     :param separator: The separator character for formatting the row.
     :type separator: str

     :param end: The end character for formatting the row.
     :type end: str

     :param markers: A list of markers to use for different dataset or method groups.
     :type markers: list[str]

     :param verbose: Whether to print verbose output. True to print, False to suppress output.
     :type verbose: bool

     :param annos: A list of annotation levels or budgets.
     :type annos: list[str]

     :return: None
     """


    x_axis_length = len(xs)
    # split selected results of means and stds from raw valeus
    raw_results = np.array([
        [e for d in range(len(selected_results)) for e in selected_results[d][3 * j + 2]] #
        for j in range(x_axis_length)
    ])



    selected_results = np.array([
                        [ selected_results[d][3*j + i]  for j in range(x_axis_length) for i in range(2)]
                        for d in range(len(selected_results))
                    ])



    if xs is None:
        return
    low_quantile = np.nanquantile(selected_results, 0.25, axis=0)
    up_quantile = np.nanquantile(selected_results, 0.75, axis=0)

    means = np.nanmean(selected_results, axis=0)
    medians = np.nanmedian(selected_results, axis=0)
    stds = np.nanstd(selected_results, axis=0)



    # measure difference to previous annotation scheme
    # rotate selected results to correct position for difference
    is_not_nan = ~np.isnan(selected_results)
    x_ = np.array([list(row[-2:]) + list(row[:-2]) for row in selected_results])
    diff = selected_results - x_
    greater = np.sum(diff >= 0, axis=0)

    missing_numbers = np.count_nonzero(is_not_nan, axis=0)
    total = selected_results.shape[0]


    if short:
        ys = np.array([np.nanmean(raw_results[i]) for i in range(x_axis_length)])
        error = np.array([sem(raw_results[i], nan_policy='omit') for i in range(x_axis_length)])

        temp = [((f'{medians[2 * i]:0.02f} &' if total > 1 else '' if not np.isnan(means[2 * i]) else "N/A" ) + f' {ys[i] :0.02f} +- {means[2 * i + 1]  if mean_stds else error[i] :0.02f}'.ljust(padding) if not np.isnan(means[2 * i]) else "N/A".ljust(padding)) for i in range(x_axis_length)]
    else:
        temp = [(f'{means[2 * i] * 100:0.02f} +- {means[2 * i + 1] * 100 if mean_stds else stds[2 * i] * 100:0.02f}  [{missing_numbers[2 * i]:02d}/{total:02d}] ({medians[2 * i] * 100:0.02f},{greater[2 * i] / missing_numbers[2 * i]:0.02f}\%)'.ljust(padding) if not np.isnan(means[2 * i]) else "N/A".ljust(padding)) for i in range(x_axis_length)]

    if verbose:
        print(f"{get_show_label_name(name).ljust(first_padding)} {seperator} "
              f"{f' {seperator} '.join(temp)}"
              f"{end}")

    # check if historgram


    if mode == 'median':
        ys = np.array([medians[2 * i] for i in range(x_axis_length)])
        error_down = np.abs([low_quantile[2 * i] for i in range(x_axis_length)] - ys)
        error_up = [up_quantile[2 * i] for i in range(x_axis_length)] - ys
        error = np.array([error_down, error_up])
        # hide errors
        error = None
    elif mode == 'mean':
        ys = np.array([means[2 * i] for i in range(x_axis_length)])
        error = np.array([stds[2 * i] for i in range(x_axis_length)])
    elif mode == 'all':
        ys = np.array([selected_results[j][2 * i] for i in range(x_axis_length) for j in range(len(selected_results))])
        xs = np.array([xs[i] for i in range(x_axis_length) for j in range(len(selected_results))])
        error = None
    else:
        raise ValueError()


    cap_size = 5

    # c = colors[c_index % len(colors)]
    m = markers[l_index % len(markers)]
    l = line_styles[l_index % len(line_styles)]
    m_size = 9
    # if label is None:
    if no_line:
        if error is None:
            plt.scatter(xs, ys, c=score_color, marker=m,s=m_size)
        else:
            plt.errorbar(xs, ys, yerr=error, c=score_color, marker=m, ls='', capsize=cap_size,markersize=m_size)
    else:
        if error is None:
            plt.plot(xs, ys, c=score_color, marker=m, linestyle=l,markersize=m_size)
        else:
            plt.errorbar(xs, ys, yerr=error, c=score_color, marker=m, ls =l, capsize=cap_size,markersize=m_size)




def check_status(report:DCICReport):
    """
    Investigate the current collection of results and what data is missing and their expected runtime

    Here is an overview of its functionality:

    1. The function initializes various variables, including lists of initial values, datasets, methods, scores, expected number of GPUs, and the number of splits.

    2. It iterates over each method and dataset combination to perform the following steps:

    a. It checks for missing data and NaN values in the results. If any data is missing, it adds the corresponding experiment to the missing list. If NaN values are present, it adds the experiment to the nans list.

    b. It calculates the runtime for Phase 1 based on the files in the output folder. If the runtime is not available, it approximates it.

    c. It updates the lists of completed times for Phase 1 and Phase 2.

    d. It calculates the total runtime and remaining time based on the completed and remaining times for Phase 1 and Phase 2.

    e. It prints the runtime details for the current method and dataset combination, including average runtime, used time, remaining time, completed splits for Phase 1 and Phase 2, and the number of NaN values and missing experiments.

    3. After iterating through all method and dataset combinations, it prints the average runtime for each method and the total runtime and remaining time.


    :param report: A DCICReport object containing the results and scores
    :return: None
    """

    init_values_general = [
        (0.1, 1), (0.2, 1), (0.5, 1), (1, 1),  # semi supervised
        (1, 3), (1, 5), (1, 10),  # multiple
        (0.1, 10), (0.2, 5), (0.5, 2)  # semi-supervised + multiple
    ]

    datasets = all_datasets.split(",")
    methods = all_methods.split(",")

# only partially available methods
    only_ssl_methods = self_methods.split(".") + ssl_methods.split(".")
    only_ssl_methods.remove('pseudo')
    only_ssl_methods.remove('divide_mix')

    scores=['macro_f1','macro_acc','kl','ece','input_macro_f1','input_macro_acc','input_kl','input_ece','input_consistency','budget','w_budget']

    expected_num_gpus = 4


    splits = 3

    total_runtime = 0
    total_remaining = 0

    runtimes_methods = []
    for m in methods:
        for d in datasets:
            missing = []
            nans = []

            # exclude some init values
            if m in only_ssl_methods:
                init_values = [(0.1, 1), (0.2, 1), (0.5, 1), (1, 1)]
            else:
                init_values = init_values_general
            # runtime estimation, all values in hours
            # values taken from wandb
            runtime_1 = {'mean':3894,'mean_s2c2':4328,'pi':3720,'pi_s2c2':4129,'pseudo_ssl':3970,'pseudo_s2c2':5022,'fixmatch':16323,'fixmatch_s2c2':14441}.get(m,-3600) / 3600
            runtime_2 = {'Benthic':5/60,'CIFAR10H':20/60,'MiceBone':12/60,'Plankton':25/60,'QualityMRI':3/60,'Synthetic':30/60,'Treeversity#1':20/60,'Treeversity#6':15/60,'Turkey':10/60, 'Pig':12/60}[d]
            # counts the used times
            done_times_1 = []
            # counts completed splits
            done_times_2 = []

            # done_splits_1 = 0
            # done_splits_2 = 0
            for a in [f"{num:02d}-{percent:0.02f}" for percent,num in init_values]:
                exp = f"{d}-{m}-{a}"
                cs = report.scores.get(exp, {})

                if len(cs) == 0:
                    missing.append(exp)
                else:

                    # check all splits
                    done_times_2.append(len(report._calculate_values_from_scores('budget', cs.get('budget', []), verbose=0)))

                    for s in scores:
                        values = report._calculate_values_from_scores(s, cs.get(s, []), verbose=0)

                        mean_v = np.mean(values)
                        if np.isnan(mean_v):
                            nans.append(a)
                            break

                output_folder = "/data/output_datasets"
                phase1_folder = join(output_folder, exp)

                if os.path.exists(phase1_folder):
                    files = sorted(os.listdir(phase1_folder))

                    # update runtime for phase 1
                    temp = runtime_1
                    if temp < 0:
                        if len(files) > 2 :
                            files = files[:3]
                            # get time difference from time stamp last modified
                            # convert from seconds to hours
                            times = [
                                (os.path.getmtime(join(phase1_folder,files[i+1])) -
                                os.path.getmtime(join(phase1_folder, files[i]))) / 3600
                                for i in range(len(files)-1)
                            ]
                            times = [t for t in times if t >= 0]
                            if len(times) > 0:
                                temp = np.mean(times)
                            else:
                                temp = 1

                        else:
                            # approximate
                            temp = 1

                    for _ in files[:3]:
                        done_times_1.append(temp)



            runtime_1 = np.average(done_times_1) if len(done_times_1) > 0 else (runtime_1 if runtime_1 > 0 else 1)
            runtimes_methods.append(runtime_1)
            # print(runtime_1)
            done_time_1 = np.sum(done_times_1)
            remaining_time_1 = (len(init_values) * splits) * runtime_1 - done_time_1

            done_time_2 = np.sum(done_times_2)*runtime_2
            remaining_time_2 = (len(init_values)*splits - np.sum(done_times_2))*runtime_2

            total_runtime += done_time_1 + done_time_2
            total_remaining += remaining_time_1*runtime_1 + remaining_time_2

            print(f"{m} - {d}: Runtime: average (Used/Remaining)"
                  f" {runtime_1:0.1f} ({done_time_1:0.01f}/{remaining_time_1:0.01f}) |"
                  f" {runtime_2:0.1f} ({done_time_2:0.01f}/{remaining_time_2:0.01f}),"
                  f"Completed splits Phase1 {len(done_times_1)}/{len(init_values)*splits} Phase2 {np.sum(done_times_2)} / {len(init_values)*splits},"
                  f"Runtime Phase1 {runtime_1:0.1f}h, Phase2 {runtime_2:0.1f}h"
                  f" NaN in Exp Phase2 {len(nans)}/{len(init_values)-len(missing)} {nans if len(nans) < 3 else '[Many]'} All Missing Exp Phase2 {len(missing)}/{len(init_values)} {missing if len(missing) < 3 else '[Many]'}")

        print(f"Runtime: {m} - {np.mean(runtimes_methods):0.02f} +- {np.std(runtimes_methods):0.02f} h")
        print(f"TOTAL: Done {total_runtime:0.01f}h {total_runtime/24:0.01f}d  Remaining {total_remaining:0.01f}h {total_remaining/24:0.01f}d with multiple gpus {total_remaining / expected_num_gpus:0.01f}h {total_remaining / expected_num_gpus/24:0.01f}d\n\n")

        runtimes_methods = []


def forward(x):
    def intern(temp):
        if temp > -0.05:
            return temp
        else:
            return -0.05 + temp / 6

    return np.array(list(map(intern,x)))

def inverse(x):

    def intern(temp):
        if temp > -0.05:
            return temp
        else:
            return (temp+0.05) * 6

    return np.array(list(map(intern,x)))

_custom_scale = (forward,inverse)

def main(argv):

    font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)

    plt.style.use('seaborn-whitegrid')



    # show the initial report

    report = DCICReport()
    report.load_results()
    # optional to show overview about all results
    # report.show()

    # show latex version of report
    first_padding = 25
    padding = 19
    keys = report.cs.keys()

    # optional investigate run time and missing data
    # check_status(report)

    # create the individual results
    # e.g.

    analyze(report, "1_more_annotations-0",
            dataset=grouped_datasets,
            method=grouped_methods,
            annos="01-0.10,01-0.20,01-0.50,01-1.00,03-1.00,05-1.00,10-1.00",
            score_names=['kl', 'ece', 'macro_acc', 'macro_f1', 'input_consistency'],
            value_normalization='absolute', x_axis='budget', y_axis='metric',
            markers=[''])



    # create interactive loop for faster graphic generation
    running = True
    while running:

        params = input("define next visualization with name/dataset/method/annos/score_name/[Options] or abort")

        if params == "abort":
            running = False
            continue

        try:

            params = params.split("/")

            if len(params) != 6:
                print("please check the input pattern!")
                continue

            # replace elements if given
            dict_replacements = {'all':all_datasets,'grouped':grouped_datasets,'easy':easy_datasets,'diff':diff_datasets}
            params[1] = dict_replacements.get(params[1], params[1])

            dict_replacements = {'ssl_methods': ssl_methods, 'self_methods': self_methods, 'noise_methods': noise_methods,
                                 'soft_methods': soft_methods, 'hard_methods':hard_methods, 'old_methods':old_methods,
                                 'grouped':grouped_methods, 'all':all_methods,}
            params[2] = dict_replacements.get(params[2], params[2])

            dict_replacements = {'full': "01-0.10,10-0.10,01-0.20,03-0.20,05-0.20,10-0.20,01-1.00,03-1.00,05-1.00,10-1.00,100-1.00",
                                 'supervised': "01-1.00,03-1.00,05-1.00,10-1.00,100-1.00",
                                 'semi': "01-0.10,10-0.10,01-0.20,03-0.20,05-0.20,10-0.20",}
            params[3] = dict_replacements.get(params[3], params[3])

            # add options
            options = {'value_normalization':'absolute', 'x_axis':'budget', 'y_axis':'method', 'mean_stds':True, 'short':False}
            splitted_options = params[5].split(",")
            for so in splitted_options:
                if "=" in so:
                    temp_splited_param = so.split("=")
                    options[temp_splited_param[0]] = temp_splited_param[1]
                else:
                    print("optional options have to be provided with name=value,name=value,...")



            analyze(report, params[0],
                    dataset=params[1],
                    method=params[2],
                    annos=params[3],
                    score_names=[params[4]],
                    **options
                    )

        except Exception as e:
            print(e)






if __name__ == '__main__':
    app.run(main)