from datasets import (
    set_hyperparameters,
    prepare_permuted_mnist_tasks,
    prepare_split_cifar100_tasks,
    prepare_split_cifar100_tasks_aka_FeCAM,
    prepare_split_mnist_tasks,
)
from main import (
    calculate_accuracy,
    get_number_of_batch_normalization_layer,
    load_pickle_file,
    set_seed,
)
from IntervalNets.interval_MLP import IntervalMLP
from IntervalNets.hmlp_ibp import HMLP_IBP
from VanillaNets.ResNet18 import ResNetBasic
from IntervalNets.interval_ZenkeNet64 import IntervalZenkeNet
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from numpy.testing import assert_almost_equal
from collections import defaultdict
from sklearn.manifold import TSNE


def load_dataset(dataset, path_to_datasets, hyperparameters):
    if dataset == "PermutedMNIST":
        return prepare_permuted_mnist_tasks(
            path_to_datasets,
            hyperparameters["shape"],
            hyperparameters["number_of_tasks"],
            hyperparameters["padding"],
            hyperparameters["no_of_validation_samples"],
        )
    elif dataset == "CIFAR100":
        return prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
        )
    elif dataset == "SplitMNIST":
        return prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
            number_of_tasks=hyperparameters["number_of_tasks"],
        )
    elif dataset == "CIFAR100_FeCAM_setup":
        return prepare_split_cifar100_tasks_aka_FeCAM(
            path_to_datasets,
            number_of_tasks=hyperparameters["number_of_tasks"],
            no_of_validation_samples_per_class=hyperparameters[
                "no_of_validation_samples_per_class"
            ],
            use_augmentation=hyperparameters["augmentation"],
        )
    else:
        raise ValueError("This dataset is currently not handled!")


def prepare_target_network(hyperparameters, output_shape):
    if hyperparameters["target_network"] == "MLP":
        target_network = IntervalMLP(
            n_in=hyperparameters["shape"],
            n_out=output_shape,
            hidden_layers=hyperparameters["target_hidden_layers"],
            use_bias=hyperparameters["use_bias"],
            no_weights=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ResNet":
        if hyperparameters["dataset"] == "TinyImageNet" or hyperparameters["dataset"] == "SubsetImageNet":
            mode = "tiny"
        elif hyperparameters["dataset"] == "CIFAR100" or hyperparameters["dataset"] == "CIFAR100_FeCAM_setup":
            mode = "cifar"
        else:
            mode = "default"
        target_network = ResNetBasic(
            in_shape=(hyperparameters["input_shape"], hyperparameters["input_shape"], 3),
                use_bias=False,
                use_fc_bias=hyperparameters["use_bias"],
                bottleneck_blocks=False,
                num_classes=output_shape,
                num_feature_maps=[16, 16, 32, 64, 128],
                blocks_per_group=[2, 2, 2, 2],
                no_weights=False,
                use_batch_norm=hyperparameters["use_batch_norm"],
                projection_shortcut=True,
                bn_track_stats=False,
                cutout_mod=True,
                mode=mode,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ZenkeNet":
        if hyperparameters["dataset"] in ["CIFAR100", "CIFAR100_FeCAM_setup"]:
            architecture = "cifar"
        elif hyperparameters["dataset"] == "TinyImageNet":
            architecture = "tiny"
        else:
            raise ValueError("This dataset is currently not implemented!")
        target_network = IntervalZenkeNet(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
            num_classes=output_shape,
            arch=architecture,
            no_weights=False,
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    return target_network


def prepare_and_load_weights_for_models(
    path_to_stored_networks,
    path_to_datasets,
    number_of_model,
    dataset,
    seed,
):
    """
    Prepare hypernetwork and target network and load stored weights
    for both models. Also, load experiment hyperparameters.

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path for all models
                                  located in subfolders
       *number_of_model*: (int) a number of the currently loaded model
       *dataset*: (string) the name of the currently analyzed dataset,
                           one of the followings: 'PermutedMNIST',
                           'SplitMNIST', 'CIFAR100' or 'CIFAR100_FeCAM_setup'
       *seed*: (int) defines a seed value for deterministic calculations
    
    Returns a dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_network_weights*: loaded weights for the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters
    """
    assert dataset in [
        "PermutedMNIST",
        "SplitMNIST",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet"
    ]
    path_to_model = f"{path_to_stored_networks}{number_of_model}/"
    hyperparameters = set_hyperparameters(dataset, grid_search=False)
    
    set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(
        dataset, path_to_datasets, hyperparameters
    )
    if hyperparameters["dataset"] == "SubsetImageNet":
        output_shape = dataset_tasks_list[0]._data["num_classes"]
    else:
        output_shape = list(
            dataset_tasks_list[0].get_train_outputs())[0].shape[0]

    # Build target network
    target_network = prepare_target_network(hyperparameters, output_shape)

    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP_IBP(
            target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    # Load weights
    hnet_weights = load_pickle_file(
        f"{path_to_model}hypernetwork_"
        f'after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    target_weights = load_pickle_file(
        f"{path_to_model}target_network_after_"
        f'{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    # Check whether the number of target weights is exactly the same like
    # the loaded weights
    for prepared, loaded in zip(
        [hypernetwork, target_network],
        [hnet_weights, target_weights],
    ):
        no_of_loaded_weights = 0
        for item in loaded:
            no_of_loaded_weights += item.shape.numel()
        assert prepared.num_params == no_of_loaded_weights
    return {
        "list_of_CL_tasks": dataset_tasks_list,
        "hypernetwork": hypernetwork,
        "hypernetwork_weights": hnet_weights,
        "target_network": target_network,
        "target_network_weights": target_weights,
        "hyperparameters": hyperparameters,
    }


def calculate_hypernetwork_output(
    target_network,
    hyperparameters,
    path_to_stored_networks,
    no_of_task_for_loading,
    no_of_task_for_evaluation,
    forward_transfer=False,
):
    
    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP_IBP(
            target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
        random_hypernetwork = deepcopy(hypernetwork)
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f"after_{no_of_task_for_loading}_task.pt"
    )
    if forward_transfer:
        assert (no_of_task_for_loading + 1) == no_of_task_for_evaluation
        no_of_task_for_evaluation = no_of_task_for_loading
        # Also embedding from the 'no_of_task_for_loading' will be loaded
        # because embedding from the foregoing task is built randomly
        # (not from zeros!)
    hypernetwork_output = hypernetwork.forward(
        cond_id=no_of_task_for_evaluation, weights=hnet_weights
    )
    random_hypernetwork_output = random_hypernetwork.forward(
        cond_id=no_of_task_for_evaluation
    )
    return random_hypernetwork_output, hypernetwork_output



def calculate_backward_transfer(dataframe):
    """
    Calculate backward transfer based on dataframe with results
    containing columns: 'loaded_task', 'evaluated_task',
    'loaded_accuracy' and 'random_net_accuracy'.
    ---
    BWT = 1/(N-1) * sum_{i=1}^{N-1} A_{N,i} - A_{i,i}
    where N is the number of tasks, A_{i,j} is the result
    for the network trained on the i-th task and tested
    on the j-th task.

    Returns a float with backward transfer result.
    """
    backward_transfer = 0
    number_of_last_task = int(dataframe.max()["loaded_task"])
    # Indeed, number_of_last_task represents the number of tasks - 1
    # due to the numeration starting from 0
    for i in range(number_of_last_task + 1):
        trained_on_last_task = dataframe.loc[
            (dataframe["loaded_task"] == number_of_last_task)
            & (dataframe["evaluated_task"] == i)
        ]["loaded_accuracy"].values[0]
        trained_on_the_same_task = dataframe.loc[
            (dataframe["loaded_task"] == i) & (dataframe["evaluated_task"] == i)
        ]["loaded_accuracy"].values[0]
        backward_transfer += trained_on_last_task - trained_on_the_same_task
    backward_transfer /= number_of_last_task
    return backward_transfer


def calculate_forward_transfer(dataframe):
    """
    Calculate forward transfer based on dataframe with results
    containing columns: 'loaded_task', 'evaluated_task',
    'loaded_accuracy' and 'random_net_accuracy'.
    ---
    FWT = 1/(N-1) * sum_{i=1}^{N-1} A_{i-1,i} - R_{i}
    where N is the number of tasks, A_{i,j} is the result
    for the network trained on the i-th task and tested
    on the j-th task and R_{i} is the result for a random
    network evaluated on the i-th task.

    Returns a float with forward transfer result.
    """
    forward_transfer = 0
    number_of_tasks = int(dataframe.max()["loaded_task"] + 1)
    for i in range(1, number_of_tasks):
        extracted_result = dataframe.loc[
            (dataframe["loaded_task"] == (i - 1))
            & (dataframe["evaluated_task"] == i)
        ]
        trained_on_previous_task = extracted_result["loaded_accuracy"].values[0]
        random_network_result = extracted_result["random_net_accuracy"].values[
            0
        ]
        forward_transfer += trained_on_previous_task - random_network_result
    forward_transfer /= number_of_tasks - 1
    return forward_transfer


def calculate_FWT_BWT_different_files(paths, forward=True):
    """
    Calculate mean forward and (or) backward transfer with corresponding
    sample standard deviations based on results saved in .csv files

    Argument:
    ---------
      *paths* (list) contains path to the results files
      *forward* (optional Boolean) defines whether forward transfer will
                be calculated
    Returns:
    --------
      *FWTs* (list of floats) contains consecutive forward transfer values
             or an empty list (if forward is False)
      *BWTs* (list of floats) contains consecutive backward transfer values
    """
    FWTs, BWTs = [], []
    for path in paths:
        dataframe = pd.read_csv(path, sep=";", index_col=0)
        if forward:
            FWTs.append(calculate_forward_transfer(dataframe))
        BWTs.append(calculate_backward_transfer(dataframe))
    if forward:
        print(
            f"Mean forward transfer: {np.mean(FWTs)}, "
            f"population standard deviation: {np.std(FWTs)}"
        )
    print(
        f"Mean backward transfer: {np.mean(BWTs)}, "
        f"population standard deviation: {np.std(BWTs)}"
    )
    return FWTs, BWTs


def evaluate_target_network(
    target_network, network_input, weights, target_network_type, condition=None
):
    """
       *condition* (optional int) the number of the currently tested task
                   for batch normalization

    Returns logits
    """
    if target_network_type == "ResNet":
        assert condition is not None
    if target_network_type == "ResNet":
        # Only ResNet needs information about the currently tested task
        return target_network.forward(
            network_input, weights=weights, condition=condition
        )
    else:
        return target_network.forward(network_input, weights=weights)

def plot_accuracy_one_setting(
    path_to_stored_networks,
    no_of_models_for_loading,
    suffix,
    dataset_name,
    folder="./Plots/",
):
    """
    Plot average accuracy for the best setting of the selected method
    for a given dataset. On the plot results after the training of models
    for all tasks are compared with the corresponding results just after
    the training of models.

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path to the folder with results
                                  for all models
       *no_of_models_for_loading*: (list) contains names of subfolders
                                   with consecutive models
       *suffix*: (string) name of the file with results; single files
                 are located in consecutive subfolders
       *dataset_name*: (string) name of the currently analyzed dataset
       *folder*: (optional string) name of the folder for saving results
    """
    individual_results_just_after, individual_results_after_all = [], []
    # Load results for all models: results after learning of all tasks
    # as well as just after learning consecutive tasks
    for model in no_of_models_for_loading:
        accuracy_results = pd.read_csv(
            f"{path_to_stored_networks}{model}/{suffix}", sep=";", index_col=0
        )
        just_after_training = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results["tested_task"]
        ]
        after_all_training_sessions = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results.max()["after_learning_of_task"]
        ]
        individual_results_just_after.append(just_after_training)
        individual_results_after_all.append(after_all_training_sessions)
    dataframe_just_after = pd.concat(
        individual_results_just_after, ignore_index=True, axis=0
    )
    dataframe_just_after["after_learning_of_task"] = "just after training"
    dataframe_after_all = pd.concat(
        individual_results_after_all, ignore_index=True, axis=0
    )
    dataframe_after_all[
        "after_learning_of_task"
    ] = "after training of all tasks"
    dataframe = pd.concat(
        [dataframe_just_after, dataframe_after_all], axis=0, ignore_index=True
    )
    dataframe = dataframe.rename(
        columns={"after_learning_of_task": "evaluation"}
    )
    dataframe["tested_task"] += 1
    tasks = individual_results_just_after[0]["tested_task"].values + 1
    ax = sns.relplot(
        data=dataframe,
        x="tested_task",
        y="accuracy",
        kind="line",
        hue="evaluation",
        height=3,
        aspect=1.5,
    )
    # mean and 95% confidence intervals
    if dataset_name in [
        "Permuted MNIST (10 tasks)",
        "Split MNIST",
        "CIFAR-100 (ResNet)",
        "CIFAR-100 (ZenkeNet)",
    ]:
        ax.set(xticks=tasks, xlabel="Number of task", ylabel="Accuracy [%]")
        legend_fontsize = 11
        if dataset_name == "Permuted MNIST (10 tasks)":
            legend_position = "upper right"
            bbox_position = (0.65, 0.95)
        else:
            legend_position = "lower center"
            bbox_position = (0.5, 0.17)
            if dataset_name == "Split MNIST":
                legend_fontsize = 10
    elif dataset_name == "Permuted MNIST (100 tasks)":
        legend_position = "lower right"
        bbox_position = (0.65, 0.2)
        tasks = np.arange(0, 101, step=10)
        tasks[0] = 1
        ax.set(xticks=tasks, xlabel="Number of task", ylabel="Accuracy [%]")
    else:
        raise ValueError("Not implemented dataset!")
    sns.move_legend(
        ax,
        legend_position,
        bbox_to_anchor=bbox_position,
        fontsize=legend_fontsize,
        title="",
    )
    plt.title(f"Results for {dataset_name}", fontsize=11)
    plt.xlabel("Number of task", fontsize=11)
    plt.ylabel("Accuracy [%]", fontsize=11)
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}mean_accuracy_best_setting_{dataset_name}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_single_model_accuracy_one_setting(
    path_to_stored_networks,
    no_of_models_for_loading,
    suffix,
    dataset_name,
    folder="./Plots/",
    legend=True,
):
    """
    Plot accuracy just after training of consecutive tasks and after
    training of all tasks for a single model of the selected method
    for a given dataset.
    This function is especially prepared for TinyImageNet

    Arguments:
    ----------
       *path_to_stored_networks*: (string) path to the folder with results
                                  for all models
       *no_of_models_for_loading*: (list) contains names of subfolders
                                   with consecutive models
       *suffix*: (string) name of the file with results; single files
                 are located in consecutive subfolders
       *dataset_name*: (string) name of the currently analyzed dataset
       *folder*: (optional string) name of the folder for saving results
       *legend*: (optional Boolean value) defines whether a legend should
                 be inserted
    """
    name_to_save = (
        dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    )
    for model in no_of_models_for_loading:
        accuracy_results = pd.read_csv(
            f"{path_to_stored_networks}{model}/{suffix}", sep=";", index_col=0
        )
        just_after_training = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results["tested_task"]
        ].copy()
        just_after_training.reset_index(inplace=True, drop=True)
        just_after_training.loc[
            :, "after_learning_of_task"
        ] = "just after training"

        after_all_training_sessions = accuracy_results.loc[
            accuracy_results["after_learning_of_task"]
            == accuracy_results.max()["after_learning_of_task"]
        ].copy()
        after_all_training_sessions.reset_index(inplace=True, drop=True)
        after_all_training_sessions.loc[
            :, "after_learning_of_task"
        ] = "after training of all tasks"
        dataframe = pd.concat(
            [just_after_training, after_all_training_sessions],
            axis=0,
            ignore_index=True,
        )
        dataframe = dataframe.rename(
            columns={"after_learning_of_task": "evaluation"}
        )
        dataframe["tested_task"] += 1
        if "ImageNet" in dataset_name:
            tasks = [0, 4, 9, 14, 19, 24, 29, 34, 39]
        else:
            tasks = just_after_training["tested_task"].values + 1
        values = dataframe["accuracy"].values
        plt.figure(figsize=(5.5, 2.5))
        ax = sns.barplot(
            data=dataframe,
            x="tested_task",
            y="accuracy",
            hue="evaluation",
        )
        ax.set(
            xticks=tasks,
            ylim=(np.min(values) - 3,
                  np.max(values) + 3)
        )
        if legend:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(0, 1.3),
                fontsize=10,
                ncol=2,
                title="",
            )
        else:
            ax._legend.remove()
            plt.title(f"Results for {dataset_name}", fontsize=10, pad=20)
        plt.xlabel("Number of task", fontsize=10)
        plt.ylabel("Accuracy [%]", fontsize=10)
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(
            f"{folder}accuracy_model_{model}_{name_to_save}.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def prepare_tSNE_plot(
    features, gt_classes, name, dataset, label="task", title=None
):
    """
    Prepare a t-SNE plot to produce an embedded version of features.

    Arguments:
    ----------
      *features* -
    """
    if dataset == "PermutedMNIST":
        fig, ax = plt.subplots(figsize=(4, 4))
        s = 0.1
        alpha = None
        legend_loc = "best"
        bbox_to_anchor = None
        fontsize = 9
        legend_fontsize = "medium"
        legend_titlefontsize = None
    elif dataset == "SplitMNIST":
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        s = 0.5
        alpha = 0.75
        legend_loc = "center"
        bbox_to_anchor = (1.15, 0.5)
        fontsize = 8
        legend_fontsize = "small"
        legend_titlefontsize = "small"
    # legend position outside for Split
    values = np.unique(gt_classes)
    for i in values:
        plt.scatter(
            features[gt_classes == i, 0],
            features[gt_classes == i, 1],
            label=i,
            rasterized=True,
            s=s,
            alpha=alpha,
        )
    lgnd = plt.legend(
        title=label,
        loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=legend_fontsize,
        title_fontsize=legend_titlefontsize,
        handletextpad=0.1,
    )
    for i in range(values.shape[0]):
        lgnd.legendHandles[i]._sizes = [20]
    plt.title(title, fontsize=fontsize)
    plt.xlabel("t-SNE embedding first dimension", fontsize=fontsize)
    plt.ylabel("t-SNE embedding second dimension", fontsize=fontsize)
    plt.savefig(f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_calculate_transfers():
    """
    Unittest of calculate_backward_transfer() and calculate_forward_transfer()
    """
    test_results_1 = [
        [0, 0, 80, 15],
        [0, 1, 20, 15],
        [0, 2, 13, 16],
        [0, 3, 19, 18],
        [1, 0, 35, 17],
        [1, 1, 85, 10],
        [1, 2, 20, 18],
        [1, 3, 18, 15],
        [2, 0, 30, 12],
        [2, 1, 10, 15],
        [2, 2, 70, 16],
        [2, 3, 25, 17],
        [3, 0, 35, 17],
        [3, 1, 40, 21],
        [3, 2, 25, 15],
        [3, 3, 90, 10],
    ]
    test_dataframe_1 = pd.DataFrame(
        test_results_1,
        columns=[
            "loaded_task",
            "evaluated_task",
            "loaded_accuracy",
            "random_net_accuracy",
        ],
    )
    output_BWT_1 = calculate_backward_transfer(test_dataframe_1)
    gt_BWT_1 = -45
    assert_almost_equal(output_BWT_1, gt_BWT_1)
    output_FWT_1 = calculate_forward_transfer(test_dataframe_1)
    gt_FWT_1 = 5
    assert_almost_equal(output_FWT_1, gt_FWT_1)


if __name__ == "__main__":
   
   pass