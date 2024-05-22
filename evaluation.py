from datasets import (
    set_hyperparameters,
    prepare_permuted_mnist_tasks,
    prepare_split_cifar100_tasks,
    prepare_split_cifar100_tasks_aka_FeCAM,
    prepare_split_mnist_tasks,
)
from main import (
    load_pickle_file,
    set_seed,
    intersection_of_embeds
)
from IntervalNets.interval_MLP import IntervalMLP
from IntervalNets.hmlp_ibp import HMLP_IBP
from VanillaNets.ResNet18 import ResNetBasic
from IntervalNets.interval_ZenkeNet64 import IntervalZenkeNet
from copy import deepcopy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from typing import Tuple

def load_dataset(dataset, path_to_datasets, hyperparameters):
    if dataset == "PermutedMNIST":
        return prepare_permuted_mnist_tasks(
            path_to_datasets,
            hyperparameters["shape"],
            hyperparameters["number_of_tasks"],
            hyperparameters["padding"],
            hyperparameters["no_of_validation_samples"],
        )
    elif dataset == "CIFAR-100":
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
        elif hyperparameters["dataset"] == "CIFAR-100" or hyperparameters["dataset"] == "CIFAR100_FeCAM_setup":
            mode = "cifar"
        else:
            mode = "default"
        target_network = ResNetBasic(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
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
        if hyperparameters["dataset"] in ["CIFAR-100", "CIFAR100_FeCAM_setup"]:
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
                           'SplitMNIST', 'CIFAR-100' or 'CIFAR100_FeCAM_setup'
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
        "SubsetImageNet",
        "CIFAR10"
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

    perturbation_vectors = load_pickle_file(
        f"{path_to_model}perturbation_vectors_after_"
        f'{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )

    hypernetwork._perturbated_eps_T = perturbation_vectors

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

def plot_accuracy_curve(
        list_of_folders_path,
        save_path,
        filename,
        mode = 1,
        perturbation_sizes = [5.0, 10.0, 15.0, 20.0, 25.0],
        dataset_name = "PermutedMNIST-10",
        beta_params = [1.0, 0.1, 0.05, 0.01, 0.001],
        y_lim_max = 100.0,
        fontsize = 10,
        figsize = (6, 4)
):
    """
        This function saves the accuracy curve for the specified mode

        Arguments:
        ---------
            *list_of_folders_path*: (List[str]) a list with paths to stored results,
                                    one path for one seed
            *save_path*: (str) the path where plots will be stored
            *filename*: (str) name of saved plot
            *mode*: (int) an integer having the following meanings:
                - 1: experiments for different perturbation size for the
                     universal embedding setup
                - 2: experiments for different beta parameters for the
                     universal embedding setup
                - 3: experiments for different perturbation size for the
                     non-forced setup
                - 4: experiments for different beta parameters for the
                     non-forced setup
            *perturbation_sizes*: (list) a list with gamma hyperparameters
            *dataset_name*: (str) a dataset name
            *beta_params*: (list) a list with beta hyperparameters,
            *y_lim_max*: (float) an upper limit of y axis
            *fontsize*: (int) font size of the titles and axes
            *figsize*: (tuple[int]) a tuple with width and height of the
                       figures
        
        Returns:
        --------
            None
    """

    assert mode in [1, 2, 3, 4], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    results_folder_seed_1 = os.listdir(list_of_folders_path[0])
    results_folder_seed_2 = os.listdir(list_of_folders_path[1])

    title = f'Results for {dataset_name}'

    if mode in [1,3]:
        params = perturbation_sizes
        title = f'{title} for $\gamma$ hyperparameters'
    elif mode in [2,4]:
        params = beta_params
        title = f'{title} for $\\beta$ hyperparameters'

    dataframe = {}
    fig, ax = plt.subplots(figsize=figsize)
    
    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i+1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i+1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i+1 for i in range(40)]
    

    for (results_seed_1, results_seed_2, param) in zip(
                                    results_folder_seed_1, 
                                    results_folder_seed_2, 
                                    params):
        if mode in [1, 2]:
            acc_path_1 = f'{list_of_folders_path[0]}/{results_seed_1}/results_intersection.csv'
            acc_path_2 = f'{list_of_folders_path[1]}/{results_seed_2}/results_intersection.csv'
        else:
            acc_path_1 = f'{list_of_folders_path[0]}/{results_seed_1}/results.csv'
            acc_path_2 = f'{list_of_folders_path[1]}/{results_seed_2}/results.csv'

        pd_results_1 = pd.read_csv(acc_path_1, sep=";")
        pd_results_2 = pd.read_csv(acc_path_2, sep=";")

        acc_1 = pd_results_1.groupby(["after_learning_of_task"]).mean()
        acc_2 = pd_results_2.groupby(["after_learning_of_task"]).mean()

        acc_1 = acc_1["accuracy"]
        acc_2 = acc_2["accuracy"]

        dataframe["accuracy"] = (acc_1 + acc_2)/2.0

        if "beta" in title:
            ax.plot(tasks_list, dataframe["accuracy"], label=f"$\\beta = {param}$")
        elif "gamma" in title:
            ax.plot(tasks_list, dataframe["accuracy"], label=f"$\gamma = {param}$")
        
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    ax.set_xticks(range(1, tasks_list[-1]+1))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()


def plot_accuracy_curve_for_diff_nesting_methods(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name = "PermutedMNIST-10",
        y_lim_max = 100.0,
        figsize = (8, 4),
        fontsize = 10
):
    """
        This function saves the accuracy curve for the different
        nesting methods, such as tanh or cosine

        Arguments:
        ---------
            *list_of_folders_path*: (List[str]) a list with paths to stored results,
                                    the first two paths are for the tanh nesting method and
                                    the last two are for the cosine nesting method
            *save_path*: (str) the path where plots will be stored
            *filename*: (str) name of saved plot
            *dataset_name*: (str) a dataset name
            *y_lim_max*: (float) an upper limit of y axis
            *figsize*: (tuple of ints), figure size
            *fontsize*: (int)
        
        Returns:
        --------
            None
    """

    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Nesting results for {dataset_name}'

    tanh_dataframe = {}
    cos_dataframe = {}
    fig, ax = plt.subplots()
    
    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i+1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i+1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i+1 for i in range(40)]
    
    tanh_acc_path_1 = f'{list_of_folders_path[0]}/results_intersection.csv'
    tanh_acc_path_2 = f'{list_of_folders_path[1]}/results_intersection.csv'

    cos_acc_path_1 = f'{list_of_folders_path[2]}/results_intersection.csv'
    cos_acc_path_2 = f'{list_of_folders_path[3]}/results_intersection.csv'

    tanh_pd_results_1 = pd.read_csv(tanh_acc_path_1, sep=";")
    tanh_pd_results_2 = pd.read_csv(tanh_acc_path_2, sep=";")

    cos_pd_results_1 = pd.read_csv(cos_acc_path_1, sep=";")
    cos_pd_results_2 = pd.read_csv(cos_acc_path_2, sep=";")

    tanh_acc_1 = tanh_pd_results_1.groupby(["after_learning_of_task"]).mean()
    tanh_acc_2 = tanh_pd_results_2.groupby(["after_learning_of_task"]).mean()

    cos_acc_1 = cos_pd_results_1.groupby(["after_learning_of_task"]).mean()
    cos_acc_2 = cos_pd_results_2.groupby(["after_learning_of_task"]).mean()

    tanh_acc_1 = tanh_acc_1["accuracy"]
    tanh_acc_2 = tanh_acc_2["accuracy"]

    cos_acc_1 = cos_acc_1["accuracy"]
    cos_acc_2 = cos_acc_2["accuracy"]

    tanh_dataframe["accuracy"] = (tanh_acc_1 + tanh_acc_2)/2.0
    cos_dataframe["accuracy"] = (cos_acc_1 + cos_acc_2)/2.0

    ax.plot(tasks_list, tanh_dataframe["accuracy"], label=f"$\\tanh$")
    ax.plot(tasks_list, cos_dataframe["accuracy"], label=f"$\cos$")
        
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    ax.set_xticks(range(1, tasks_list[-1]+1))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_histogram_of_intervals(path_to_stored_networks,
                                save_path,
                                filename,
                                parameters,
                                num_bins: int = 10,
                                threshold_collapsed: float = 1e-8,
                                density: bool = True,
                                plot_vlines: bool = True,
                                figsize: Tuple[int] = (7,5),
                                rotation = False,
                                fontsize = 10
                                ):
    """
    Arguments:
    ----------
        *path_to_stored_networks*: str, path to folder where the saved hypernetwork is stored
        *save_path*: str, path to folder, where the histogram will be saved
        *filename*: str, a name of the saved histogram plot
        *parameters*: dict, a dictionary with the following hyperparameters:
            - number_of_tasks - int, a number of CL tasks
            - perturbated_epsilon - float, a perturbation value
            - embedding_size - int, dimensionality of the embedding space
            - activation_function - nn.Module, a non-linear non-decreasing activation function
            - hypernetwork_hidden_layers - list of integers indicating a number of hidden layers
                                           and neuron per each layer in the hypernetwork
            - shape - integer indicating width (height)
            - out_shape - number of neurons at the last layer of the neural network
            - target_hidden_layers - list of integers indicating a number of hidden layers
                                           and neuron per each layer in the target network
            - use_bias - bool, True if biased is used, False otherwise
            - use_batch_norm - bool, True if batch normalization layers are used, False
                               otherwise, applicable only in the ResNet/ZenkeNet architecture
            - target_network - str, a name of the target network, e.g. MLP, ResNet or ZenkeNet
            - dataset - str, which dataset is used, it is essential to determine input shape
                        to the target network
            - device - str, `cuda` or `cpu`
        *num_bins*: int, a number of histogram bins (a number of bars),
        *threshold_collapsed*: float, a threshold for treating the interval as collapsed to a point,
        *density*: bool, if True, draws and returns a probability density,
        *plot_vlines*: bool, if True, vertical lines will be drawn to separate each bar,
        *figsize*: a tuple, represents the size of a plot,
        *path*: str, path to a saving directory,
        *rotation*: bool, if True, then OX axis ticks will be rotated by 45 degrees to the right,
        *fontsize*: int, size of font in title, OX and OY axes
    """
    os.makedirs(save_path, exist_ok=True)
    target_network = prepare_target_network(parameters, parameters["out_shape"])

    eps = parameters["perturbated_epsilon"]
    dim_emb = parameters["embedding_size"]
    no_tasks = parameters["number_of_tasks"]
    sigma = 0.5 * eps / dim_emb

    hypernetwork = HMLP_IBP(
            perturbated_eps=eps,
            target_shapes=target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=dim_emb,
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=no_tasks)
    
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f'after_{parameters["number_of_tasks"] - 1}_task.pt'
     )
    
    embds = hnet_weights[:no_tasks]

    with torch.no_grad():
        radii = torch.stack([
            eps * F.softmax(torch.ones(dim_emb), dim=-1) for i in range(no_tasks)
        ], dim=0)

        embds_centers = torch.stack([
            sigma * torch.cos(embds[i].detach()) for i in range(no_tasks)
        ], dim=0)

        universal_embedding_lower, universal_embedding_upper = intersection_of_embeds(embds_centers - radii, embds_centers + radii)

        universal_embedding = (universal_embedding_lower + universal_embedding_upper)/2.0
        universal_radii = (universal_embedding_upper - universal_embedding_lower)/2.0

        W_lower, _, W_upper, _ = hypernetwork.forward(
            cond_input = universal_embedding.view(1, -1),
            return_extended_output = True,
            weights = hnet_weights,
            common_radii = universal_radii
        )

        plt.rcParams["figure.figsize"] = figsize
        epsilon = [(upper - lower).view(-1) for upper, lower in zip(W_upper, W_lower)]
        outputs = torch.cat(epsilon)
        num_zero_outputs = torch.where(outputs < threshold_collapsed, 1, 0).sum().item()
        ylabel = "Desity of intervals" if density else "Number of intervals"

        outputs = outputs.detach().numpy()

        n, bins, patches = plt.hist(outputs,
                                    num_bins,
                                    density = density,
                                    color = "green",
                                    alpha = 0.5)

        plt.xlabel('Interval length', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        ticks = np.linspace(start=outputs.min(), stop=outputs.max(), num=num_bins+1)
        if plot_vlines:
            plt.vlines(x = ticks[:-1], ymin = 0, ymax = n, linestyle="--", color='black')
        plt.xticks(ticks)
        plt.title('Density of epsilon intervals', fontweight = "bold", fontsize=fontsize)
        if rotation:
            plt.xticks(rotation=45, ha='right')
        plt.legend([], title=f'Total number of coordinates = {len(outputs)}\nNumber of collapsed coordinates = {num_zero_outputs}', fontsize=fontsize)
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{save_path}{filename}')
        plt.close()

def plot_intervals_around_embeddings_for_trained_models(path_to_stored_networks,
                                                        save_path,
                                                        filename,
                                                        parameters,
                                                        figsize: Tuple[int] = (7,5),
                                                        rotation = False,
                                                        fontsize = 10,
                                                        dims_to_plot = 5):
    """
    Arguments:
    ----------
        *path_to_stored_networks*: str, path to folder where the saved hypernetwork is stored
        *save_path*: str, path to folder, where the plot will be saved
        *filename*: str, a filename of the saved plot
        *parameters*: dict, a dictionary with the following hyperparameters:
            - number_of_tasks - int, a number of CL tasks
            - perturbated_epsilon - float, a perturbation value
            - embedding_size - int, dimensionality of the embedding space
            - activation_function - nn.Module, a non-linear non-decreasing activation function
            - hypernetwork_hidden_layers - list of integers indicating a number of hidden layers
                                           and neuron per each layer in the hypernetwork
            - shape - integer indicating width (height)
            - out_shape - number of neurons at the last layer of the neural network
            - target_hidden_layers - list of integers indicating a number of hidden layers
                                           and neuron per each layer in the target network
            - use_bias - bool, True if biased is used, False otherwise
            - use_batch_norm - bool, True if batch normalization layers are used, False
                               otherwise, applicable only in the ResNet/ZenkeNet architecture
            - target_network - str, a name of the target network, e.g. MLP, ResNet or ZenkeNet
            - dataset - str, which dataset is used, it is essential to determine input shape
                        to the target network
            - device - str, `cuda` or `cpu`
        *figsize*: a tuple, represents the size of a plot,
        *path*: str, path to a saving directory,
        *rotation*: bool, if True, then OX axis ticks will be rotated by 45 degrees to the right,
        *fontsize*: int, size of font in title, OX and OY axes
        *dims_to_plot*: int, number of the first dimensions to plots
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    no_tasks = parameters["number_of_tasks"]
    dim_emb  = parameters["embedding_size"]
    eps      = parameters["perturbated_epsilon"]
    sigma    = 0.5 * eps / dim_emb

    target_network = prepare_target_network(parameters, parameters["out_shape"])

    hypernetwork = HMLP_IBP(
            perturbated_eps=parameters["perturbated_epsilon"],
            target_shapes=target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"])
    
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f'after_{parameters["number_of_tasks"] - 1}_task.pt'
    )

    with torch.no_grad():
        
        embds = hnet_weights[:no_tasks]

        radii = torch.stack([
            eps * F.softmax(torch.ones(dim_emb), dim=-1) for i in range(no_tasks)
        ], dim=0)

        embds_centers = torch.stack([
            sigma * torch.cos(embds[i].detach()) for i in range(no_tasks)
        ], dim=0)

        universal_embedding_lower, universal_embedding_upper = intersection_of_embeds(embds_centers - radii, embds_centers + radii)

        universal_embedding = (universal_embedding_lower + universal_embedding_upper)/2.0
        universal_radii = (universal_embedding_upper - universal_embedding_lower)/2.0

        universal_embedding = universal_embedding.cpu().detach().numpy()
        universal_radii = universal_radii.cpu().detach().numpy()

        universal_embedding = universal_embedding[:dims_to_plot]
        universal_radii = universal_radii[:dims_to_plot]
        
        # Create a plot
        fig = plt.figure(figsize=figsize)
        cm  = plt.get_cmap("gist_rainbow")

        colors = [cm(1.*i/(no_tasks + 1)) for i in range(no_tasks + 1)]

        for task_id, (tasks_embeddings, radii_per_emb) in enumerate(zip(embds_centers, radii)):
            
            tasks_embeddings = tasks_embeddings.cpu().detach().numpy()
            radii_per_emb = radii_per_emb.cpu().detach().numpy()

            tasks_embeddings = tasks_embeddings[:dims_to_plot]
            radii_per_emb = radii_per_emb[:dims_to_plot]

            # Generate an x axis
            x = [_ for _ in range(dims_to_plot)]

            # Create a scatter plot
            plt.scatter(x, tasks_embeddings, label=f"{task_id}-th task", marker="o", c=[colors[task_id]], alpha=0.3)

            for i in range(len(x)):
                plt.vlines(x[i], ymin=tasks_embeddings[i] - radii_per_emb[i],
                            ymax=tasks_embeddings[i] + radii_per_emb[i], linewidth=2, colors=[colors[task_id]], alpha=0.3)
        
        plt.scatter(x, universal_embedding, label=f"Intersection", marker="o", c=[colors[-1]], alpha=1.0)

        for i in range(len(x)):
            plt.vlines(x[i], ymin=universal_embedding[i] - universal_radii[i],
                        ymax=universal_embedding[i] + universal_radii[i], linewidth=2, colors=[colors[-1]], alpha=1.0)

        # Add labels and a legend
        plt.xlabel("Number of embedding's coordinate", fontsize=fontsize)
        plt.xticks(x, range(1, dims_to_plot+1))
        plt.ylabel("Embedding's value", fontsize=fontsize)
        plt.title(f'Intervals around embeddings', fontsize=fontsize)
        if rotation:
            plt.xticks(x, rotation = 45, ha = "right")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=fontsize)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{save_path}/{filename}.png', dpi=300)
        plt.close()

def plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name="PermutedMNIST-10",
        mode=1,
        y_lim_max=100.0,
        fontsize=10,
        figsize=(6, 4),
        legend_loc = "upper right"
):
    """
        This function saves the accuracy curve for the specified mode with 95% confidence intervals.

        Arguments:
        ---------
            *list_of_folders_path*: (List[str]) a list with paths to stored results,
                                    one path for one seed
            *save_path*: (str) the path where plots will be stored
            *filename*: (str) name of saved plot
            *dataset_name*: (str) a dataset name
            *mode*: (int):
                - 1 - results will be obtained for the non forced intervals method
                - 2 - results will be obtained for the universal embedding method
            *y_lim_max*: (float) an upper limit of the OY axis
            *fontsize*: (int) font size of the titles and axes
            *figsize*: (tuple[int]) a tuple with width and height of the
                       figures
            *legend_loc*: (str), location of the legend
        
        Returns:
        --------
            None
    """

    assert len(list_of_folders_path) == 5, "Please provide results on 5 runs!"
    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Results for {dataset_name}'

    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i + 1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i + 1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i + 1 for i in range(40)]

    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    for folder in list_of_folders_path:
        acc_path = os.path.join(folder, file_suffix)
        results_list.append(pd.read_csv(acc_path, sep=";"))

    acc_just_after_training = []
    acc_after_all_training_sessions = []

    for pd_results in results_list:
        acc_just_after_training.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["tested_task"], "accuracy"].values)
        acc_after_all_training_sessions.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["after_learning_of_task"].max(), "accuracy"].values)

    acc_just_after_training = np.array(acc_just_after_training)
    acc_after_all_training_sessions = np.array(acc_after_all_training_sessions)

    mean_just_after_training = np.mean(acc_just_after_training, axis=0)
    mean_after_all_training_sessions = np.mean(acc_after_all_training_sessions, axis=0)

    ci_just_after_training = stats.t.interval(0.95, len(acc_just_after_training) - 1, loc=mean_just_after_training,
                                              scale=stats.sem(acc_just_after_training, axis=0))
    ci_after_all_training_sessions = stats.t.interval(0.95, len(acc_after_all_training_sessions) - 1,
                                                      loc=mean_after_all_training_sessions,
                                                      scale=stats.sem(acc_after_all_training_sessions, axis=0))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(tasks_list, mean_just_after_training, label="Just after training")
    ax.fill_between(tasks_list, ci_just_after_training[0], ci_just_after_training[1], alpha=0.2)
    ax.plot(tasks_list, mean_after_all_training_sessions, label="After training of all tasks")
    ax.fill_between(tasks_list, ci_after_all_training_sessions[0], ci_after_all_training_sessions[1], alpha=0.2)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    ax.set_xticks(range(1, tasks_list[-1] + 1))
    ax.set_ylim(top=y_lim_max)
    ax.legend(loc=legend_loc, fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_accuracy_curve_with_barplot(
        folder_path,
        save_path,
        filename,
        dataset_name="TinyImageNet",
        mode=1,
        y_lim_max=100.0,
        fontsize=10,
        figsize=(6, 4),
        bar_width = 0.35
):
    """
        This function saves the accuracy curve as a bar plot for the specified mode.

        Arguments:
        ---------
            *list_of_folders_path*: (List[str]) a list with paths to stored results,
                                    one path for one seed
            *save_path*: (str) the path where plots will be stored
            *filename*: (str) name of saved plot
            *dataset_name*: (str) a dataset name
            *mode*: (int):
                - 1 - results will be obtained for the non forced intervals method
                - 2 - results will be obtained for the universal embedding method
            *y_lim_max*: (float) an upper limit of the OY axis
            *fontsize*: (int) font size of the titles and axes
            *figsize*: (tuple[int]) a tuple with width and height of the
                       figures
            *bar_width*: (float)
        
        Returns:
        --------
            None
    """

    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Results for {dataset_name}'

    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i + 1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i + 1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i + 1 for i in range(40)]

    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    acc_path = os.path.join(folder_path, file_suffix)
    results_list.append(pd.read_csv(acc_path, sep=";"))

    acc_just_after_training = []
    acc_after_all_training_sessions = []

    for pd_results in results_list:
        acc_just_after_training.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["tested_task"], "accuracy"].values)
        acc_after_all_training_sessions.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["after_learning_of_task"].max(), "accuracy"].values)

    acc_just_after_training = np.array(acc_just_after_training)
    acc_after_all_training_sessions = np.array(acc_after_all_training_sessions)

    mean_just_after_training = np.mean(acc_just_after_training, axis=0)
    mean_after_all_training_sessions = np.mean(acc_after_all_training_sessions, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    bar_positions = np.arange(len(tasks_list))

    ax.bar(bar_positions - bar_width / 2, mean_just_after_training, bar_width, label="Just after training")
    ax.bar(bar_positions + bar_width / 2, mean_after_all_training_sessions, bar_width, label="After training of all tasks")

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)

    # Set x-ticks at every 5 tasks
    ax.set_xticks(np.arange(0, len(tasks_list), 5))
    ax.set_xticklabels(np.arange(1, len(tasks_list) + 1, 5))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper right", fontsize=fontsize)
    ax.grid(axis='y')

    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_heatmap_for_n_runs(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name="PermutedMNIST-10",
        mode=1,
        fontsize=10,
):
    """
        This function saves the accuracy curve for the specified mode with 95% confidence intervals.

        Arguments:
        ---------
            *list_of_folders_path*: (List[str]) a list with paths to stored results,
                                    one path for one seed
            *save_path*: (str) the path where plots will be stored
            *filename*: (str) name of saved plot
            *dataset_name*: (str) a dataset name
            *mode*: (int):
                - 1 - results will be obtained for the non forced intervals method
                - 2 - results will be obtained for the universal embedding method
            *fontsize*: (int) font size of the titles and axes
        
        Returns:
        --------
            None
    """

    assert len(list_of_folders_path) == 5, "Please provide results on 5 runs!"
    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Mean accuracy for 5 runs of HyperInterval for {dataset_name}'
    
    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    for folder in list_of_folders_path:
        acc_path = os.path.join(folder, file_suffix)
        results_list.append(pd.read_csv(acc_path, sep=";"))
    
    dataframe = pd.read_csv(acc_path, sep=";")

    acc = []

    for pd_results in results_list:
        acc.append(pd_results["accuracy"].values)

    acc = np.mean(acc, axis=0)
    dataframe["accuracy"] = acc
    dataframe = dataframe.astype(
        {"after_learning_of_task": "int32",
         "tested_task": "int32"}
        )
    table = dataframe.pivot(
        "after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.xlabel("Number of the tested task")
    plt.ylabel("Number of the previously learned task")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{filename}", dpi=300)
    plt.close()

if __name__ == "__main__":
    # ######################################################
    # # Different interval sizes - universal embedding
    # ######################################################
    # list_of_folders_path = [
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_08-45-18/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_11-51-01/"
    # ]

    # plot_accuracy_curve(
    #     list_of_folders_path,
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/interval_sizes/universal_embedding/",
    #     filename = "universal_embedding_interval_sizes.png",
    #     mode = 1,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )

    # ######################################################
    # # Different interval sizes - non forced intersections
    # ######################################################
    # list_of_folders_path = [
    #     "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_08-47-30/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_11-55-46/"
    # ]


    # plot_accuracy_curve(
    #     list_of_folders_path,
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/interval_sizes/non_forced/",
    #     filename = "non_forced_interval_sizes.png",
    #     mode = 3,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )

    # ######################################################
    # # Different regularization - universal embedding
    # ######################################################
    # list_of_folders_path = [
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_08-38-36/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_09-42-33/"
    # ]

    # plot_accuracy_curve(
    #     list_of_folders_path,
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/regularization/universal_embedding/",
    #     filename = "universal_embedding_regularization.png",
    #     mode = 2,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )

    # ######################################################
    # # Different regularization - non forced intersections
    # ######################################################
    # list_of_folders_path = [
    #    "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_22-13-42/",
    #    "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_22-16-15/"
    # ]

    # plot_accuracy_curve(
    #     list_of_folders_path,
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/regularization/non_forced/",
    #     filename = "non_forced_regularization.png",
    #     mode = 4,
    #     figsize = (8, 4),
    #     fontsize = 12,
    #     y_lim_max = 98
    # )


    # #####################################################
    # # Histograms - CIFAR100 (20 classes per 5 tasks) - nesting method
    # #####################################################
    
    # parameters = {
    #     "number_of_tasks": 5,
    #     "perturbated_epsilon": 30,
    #     "embedding_size": 48,
    #     "activation_function": torch.nn.ReLU(),
    #     "hypernetwork_hidden_layers": [100],
    #     "target_network": "ResNet",
    #     "shape": 32,
    #     "out_shape": 20,
    #     "use_bias": True,
    #     "use_batch_norm": True,
    #     "dataset": "CIFAR100_FeCAM_setup",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }  

   
    # plot_histogram_of_intervals(
    #     path_to_stored_networks = "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/CIFAR100_FeCAM_setup/val_loss/2024-05-06_10-34-20/0/",
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/histogram_results/",
    #     filename = "CIFAR100_FeCAM_weights_histogram.png",
    #     parameters = parameters,
    #     num_bins = 10,
    #     threshold_collapsed = 1e-8,
    #     rotation=True,
    #     density = True,
    #     plot_vlines = True,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )

    # ######################################################
    # # Histograms - SplitMNIST (2 classes per 5 tasks) - nesting method
    # ######################################################
    # parameters = {
    #     "number_of_tasks": 5,
    #     "perturbated_epsilon": 15,
    #     "embedding_size": 24,
    #     "activation_function": torch.nn.ReLU(),
    #     "hypernetwork_hidden_layers": [75, 75],
    #     "target_network": "MLP",
    #     "target_hidden_layers": [400, 400],
    #     "shape": 784,
    #     "out_shape": 2,
    #     "use_bias": True,
    #     "use_batch_norm": False,
    #     "dataset": "SplitMNIST",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }  

   
    # plot_histogram_of_intervals(
    #     path_to_stored_networks = "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/0/",
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/histogram_results/",
    #     filename = "SplitMNIST_weights_histogram.png",
    #     parameters = parameters,
    #     num_bins = 10,
    #     rotation = True,
    #     threshold_collapsed = 1e-8,
    #     density = True,
    #     plot_vlines = True,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )

    # ######################################################
    # # Histograms - PermutedMNIST (10 classes per 10 tasks) - nesting method
    # ######################################################
    # parameters = {
    #     "number_of_tasks": 10,
    #     "perturbated_epsilon": 5,
    #     "embedding_size": 24,
    #     "activation_function": torch.nn.ReLU(),
    #     "hypernetwork_hidden_layers": [100, 100],
    #     "target_network": "MLP",
    #     "target_hidden_layers": [1000, 1000],
    #     "shape": 784,
    #     "out_shape": 10,
    #     "use_bias": True,
    #     "use_batch_norm": False,
    #     "dataset": "PermutedMNIST",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }  

   
    # plot_histogram_of_intervals(
    #     path_to_stored_networks = "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/0/",
    #     save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/histogram_results/",
    #     filename = "PermutedMNIST_weights_histogram.png",
    #     parameters = parameters,
    #     num_bins = 50,
    #     rotation = True,
    #     threshold_collapsed = 1e-8,
    #     density = True,
    #     plot_vlines = True,
    #     figsize = (8, 4),
    #     fontsize = 12
    # )             
             

    # ######################################################
    # # Different nesting methods
    # ######################################################

    # list_of_folders_path = [
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_08-39-40/0/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-12_11-32-44/0/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/0/",
    #     "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/1/"
    # ]
    # plot_accuracy_curve_for_diff_nesting_methods(
    #    list_of_folders_path=list_of_folders_path,
    #    save_path="/home/gmkrukow/non_forced_intersections/AblationResults/nesting/",
    #    filename="tanh_cos_nesting.png",
    #    figsize = (8, 4),
    #     fontsize = 12,
    #    y_lim_max=98.0
    # )

    # ######################################################
    # # Intervals around embeddings - CIFAR100_FeCAM_setup
    # ######################################################

    # parameters = {
    #     "number_of_tasks": 5,
    #     "perturbated_epsilon": 30,
    #     "embedding_size": 48,
    #     "activation_function": torch.nn.ReLU(),
    #     "hypernetwork_hidden_layers": [100],
    #     "target_network": "ResNet",
    #     "target_hidden_layers": None,
    #     "shape": 32,
    #     "out_shape": 20,
    #     "use_bias": True,
    #     "use_batch_norm": True,
    #     "dataset": "CIFAR100_FeCAM_setup",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }  

    # save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/intervals_around_embeddings"

    # plot_intervals_around_embeddings_for_trained_models(path_to_stored_networks = "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/CIFAR100_FeCAM_setup/val_loss/2024-05-06_10-34-20/0/",
    #                                                     save_path = save_path,
    #                                                     filename = "intervals_around_embeddings_CIFAR100_FeCAM_setup",
    #                                                     parameters = parameters,
    #                                                     rotation = False,
    #                                                     figsize = (8, 4),
    #                                                     fontsize = 12,
    #                                                     dims_to_plot = 10)


    #  ######################################################
    # # Intervals around embeddings - SplitMNIST
    # ######################################################

    # parameters = {
    #     "number_of_tasks": 5,
    #     "perturbated_epsilon": 15,
    #     "embedding_size": 24,
    #     "activation_function": torch.nn.ReLU(),
    #     "hypernetwork_hidden_layers": [75, 75],
    #     "target_network": "MLP",
    #     "target_hidden_layers": [400, 400],
    #     "shape": 784,
    #     "out_shape": 2,
    #     "use_bias": True,
    #     "use_batch_norm": False,
    #     "dataset": "SplitMNIST",
    #     "device": "cuda" if torch.cuda.is_available() else "cpu"
    # }  

    # save_path = "/home/gmkrukow/non_forced_intersections/AblationResults/intervals_around_embeddings"

    # plot_intervals_around_embeddings_for_trained_models(path_to_stored_networks = "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/0/",
    #                                                     save_path = save_path,
    #                                                     filename = "intervals_around_embeddings_SplitMNIST",
    #                                                     parameters = parameters,
    #                                                     rotation = False,
    #                                                     figsize = (8, 4),
    #                                                     fontsize = 12,
    #                                                     dims_to_plot = 10)
    ######################################################
    # MAIN EXPERIMENTS SECTION
    ######################################################

    ######################################################
    # Accuracy curves with confidence intervals - PermutedMNIST-10 non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/0/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/1/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/2/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/3/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/4/"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename = "acc_PermutedMNIST10_non_forced.png",
        dataset_name = "PermutedMNIST-10",
        mode = 1,
        y_lim_max = 98.5,
        figsize = (8, 4),
        fontsize = 12
)

    ######################################################
    # Accuracy curves with confidence intervals - PermutedMNIST-10 universal embedding
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/0",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/1",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/2",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/3",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/4"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename = "acc_PermutedMNIST10_universal_embedding.png",
        dataset_name = "PermutedMNIST-10",
        mode = 2,
        y_lim_max = 100,
        figsize = (8, 4),
        fontsize = 12,
        legend_loc = "lower right"
)

    ######################################################
    # Accuracy curves with confidence intervals - SplitMNIST non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/0/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/1/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/2/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/3/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/4/"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename = "acc_SplitMNIST_non_forced.png",
        dataset_name = "SplitMNIST",
        mode = 1,
        # y_lim_max = 98.5,
        figsize = (8, 4),
        fontsize = 12,
        legend_loc = "lower right"
)

    ######################################################
    # Accuracy curves with confidence intervals - SplitMNIST universal embedding
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/0",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/1",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/2",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/3",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/4"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename = "acc_SplitMNIST_universal_embedding.png",
        dataset_name = "SplitMNIST",
        mode = 2,
        y_lim_max = 100,
        figsize = (8, 4),
        fontsize = 12,
        legend_loc = "lower right"
)

    ######################################################
    # Accuracy curves with confidence intervals - CIFAR100 (10 classes per 10 tasks) non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_14-23-22/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-55-57/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-56-30/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-57-53/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-58-11/0"

    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename = "acc_CIFAR100_non_forced.png",
        dataset_name = "CIFAR-100",
        mode = 1,
        y_lim_max = 90,
        figsize = (8, 4),
        fontsize = 12
    )

    ######################################################
    # Accuracy barplot - TinyImageNet non forced
    ######################################################

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"
    folder_path = "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/TinyImageNet/ResNet/2024-05-06_19-07-26/0/"

    plot_accuracy_curve_with_barplot(
        folder_path = folder_path,
        save_path = save_path,
        filename = "acc_TinyImageNet_non_forced.png",
        dataset_name="TinyImageNet",
        mode=1,
        y_lim_max=100.0,
        fontsize=12,
        figsize=(8, 4),
        bar_width = 0.35
    )

    ######################################################
    # Heatmap - PermutedMNIST non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/1",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/2",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/3",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_10-32-40/4",
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_heatmap_for_n_runs(
        list_of_folders_path = list_of_folders_path,
        save_path = save_path,
        filename = "heatmap_PermutedMNIST_non_forced.pdf",
        dataset_name="PermutedMNIST-10",
        mode=1,
        fontsize=12,
    )

    ######################################################
    # Heatmap - PermutedMNIST universal embedding
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/0/",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/1/",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/2/",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/3/",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/permuted_mnist_final_grid_experiments/val_loss/2024-05-04_15-15-20/4/",
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_heatmap_for_n_runs(
        list_of_folders_path = list_of_folders_path,
        save_path = save_path,
        filename = "heatmap_PermutedMNIST_universal_embedding.pdf",
        dataset_name="PermutedMNIST-10",
        mode=2,
        fontsize=12,
    )

    ######################################################
    # Heatmap - SplitMNIST non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/0/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/1/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/2/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/3/",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/split_mnist/augmented/2024-05-04_09-52-11/4/"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_heatmap_for_n_runs(
        list_of_folders_path = list_of_folders_path,
        save_path = save_path,
        filename = "heatmap_SplitMNIST_non_forced.pdf",
        dataset_name="SplitMNIST",
        mode=1,
        fontsize=12,
    )

    ######################################################
    # Heatmap - SplitMNIST universal embedding
    ######################################################
    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/0",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/1",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/2",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/3",
        "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/split_mnist/augmented/2024-05-05_10-37-59/4"
    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_heatmap_for_n_runs(
        list_of_folders_path = list_of_folders_path,
        save_path = save_path,
        filename = "heatmap_SplitMNIST_universal_embedding.pdf",
        dataset_name="SplitMNIST",
        mode=2,
        fontsize=12,
    )

     ######################################################
    # Heatmap - CIFAR100 non forced
    ######################################################

    list_of_folders_path = [
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_14-23-22/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-55-57/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-56-30/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-57-53/0",
        "/shared/results/pkrukowski/HyperIntervalResults/intersections_non_forced/grid_search_relu/CIFAR-100_single_seed/ResNet/2024-05-05_23-58-11/0"

    ]

    save_path = "/home/gmkrukow/non_forced_intersections/MainExperiments/accuracy_curves/"

    plot_heatmap_for_n_runs(
        list_of_folders_path = list_of_folders_path,
        save_path = save_path,
        filename = "heatmap_CIFAR100_non_forced.pdf",
        dataset_name="CIFAR-100",
        mode=1,
        fontsize=12,
    )

