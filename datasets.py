"""
This file implements functions to deal with datasets and tasks in continual learning

Author: Kamil Książek
"""

import os
import numpy as np
import torch
from hypnettorch.data.special import permuted_mnist
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from TinyImageNet import TinyImageNet


def generate_random_permutations(shape_of_data_instance,
                                 number_of_permutations):
    """
    Prepare a list of random permutations of the selected shape
    for continual learning tasks.

    Arguments:
    ----------
      *shape_of_data_instance*: a number defining shape of the dataset
      *number_of_permutations*: int, a number of permutations that will
                                be prepared; it corresponds to the total
                                number of tasks
      *seed*: int, optional argument, default: None
              if one would get deterministic results
    """
    list_of_permutations = []
    for _ in range(number_of_permutations):
        list_of_permutations.append(np.random.permutation(
            shape_of_data_instance))
    return list_of_permutations


def prepare_split_cifar100_tasks(datasets_folder,
                                 validation_size,
                                 use_augmentation,
                                 use_cutout=False):
    """
    Prepare a list of 10 tasks with 10 classes per each task.
    i-th task, where i in {0, 1, ..., 9} will store samples
    from classes {10*i, 10*i + 1, ..., 10*i + 9}.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    'apply_cutout' option form 'torch_input_transforms'.
    """
    handlers = []
    for i in range(0, 100, 10):
        handlers.append(SplitCIFAR100Data(
            datasets_folder,
            use_one_hot=True,
            validation_size=validation_size,
            use_data_augmentation=use_augmentation,
            use_cutout=use_cutout,
            labels=range(i, i + 10)
        ))
    return handlers


def prepare_permuted_mnist_tasks(datasets_folder,
                                 input_shape,
                                 number_of_tasks,
                                 padding,
                                 validation_size):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the PermutedMNIST dataset.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *input_shape*: (int) a number defining shape of the dataset
      *validation_size*: (int) The number of validation samples

    Returns a list of PermutedMNIST objects.
    """
    permutations = generate_random_permutations(
        input_shape,
        number_of_tasks
    )
    return permuted_mnist.PermutedMNISTList(
        permutations,
        datasets_folder,
        use_one_hot=True,
        padding=padding,
        validation_size=validation_size
    )


def prepare_split_mnist_tasks(datasets_folder,
                              validation_size,
                              use_augmentation,
                              number_of_tasks=5):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the SplitMNIST dataset. By default, it should be
    5 task containing consecutive pairs of classes:
    [0, 1], [2, 3], [4, 5], [6, 7] and [8, 9].

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (bool) defines whether dataset augmentation
                          will be applied
      *number_of_tasks* (int) a number defining the number of learning
                        tasks, by default 5.

    Returns a list of SplitMNIST objects.
    """
    return get_split_mnist_handlers(
        datasets_folder,
        use_one_hot=True,
        validation_size=validation_size,
        num_classes_per_task=2,
        num_tasks=number_of_tasks,
        use_torch_augmentation=use_augmentation
    )

def prepare_tinyimagenet_tasks(
    datasets_folder, seed,
    validation_size=250, number_of_tasks=40
    ):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the TinyImageNet dataset according to the WSN setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which TinyImageNet
                         is stored / will be downloaded 
      *seed*: (int) Necessary for the preparation of random permutation
              of the order of classes in consecutive tasks.
      *validation_size*: (optional int) defines the number of validation
                         samples in each task, by default it is 250 like
                         in the case of WSN
      *number_of_tasks*: (optional int) defines the number of continual
                         learning tasks (by default: 40)
    
    Returns a list of TinyImageNet objects.
    """
    # Set randomly the order of classes
    rng = np.random.default_rng(seed)
    class_permutation = rng.permutation(200)
    # 40 classification tasks with 5 classes in each
    handlers = []
    for i in range(0, 5 * number_of_tasks, 5):
        current_labels = class_permutation[i:(i + 5)]
        print(f'Order of classes in the current task: {current_labels}')
        handlers.append(
            TinyImageNet(
                data_path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                labels=current_labels
            )
        )
    return handlers


def set_hyperparameters(dataset,
                        grid_search=False,
                        part=0):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Arguments:
    ----------
      *dataset*: 'PermutedMNIST', 'SplitMNIST' or 'CIFAR100'
      *grid_search*: (Boolean optional) defines whether a hyperparameter
                     optimization should be performed or hyperparameters
                     for just a single run have to be returned
      *part* (only for SplitMNIST or CIFAR100!) selects a subset
             of hyperparameters for optimization (by default 0)

    Returns a dictionary with necessary hyperparameters.
    """

    if dataset == 'PermutedMNIST':
        if grid_search:
            hyperparams = {
                'embedding_sizes': [8, 16, 24],
                'learning_rates': [0.001, 0.01],
                'batch_sizes': [128],
                'betas': [0.001, 0.005, 0.0005],
                'hypernetworks_hidden_layers': [[100, 100], [25, 25]],
                'perturbated_epsilon': [10.0, 5.0, 1.0, 0.5],
                'gammas': [0.01, 0.001, 0.0005],
                'best_model_selection_method': 'val_loss',
                # not for optimization
                'seed': [1, 2, 3, 4, 5]
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams['best_model_selection_method'] = 'last_model'
            hyperparams['saving_folder'] = (
                '/shared/results/pkrukowski/HyperIntervalResults/non_forced_intervals/grid_search_relu/'
                f'permuted_mnist_final_grid_experiments/{hyperparams["best_model_selection_method"]}/'
            )

        else:
            # single run experiment
            hyperparams = {
                'seed': [3],
                'embedding_sizes': [24],
                'learning_rates': [0.001],
                'batch_sizes': [128],
                'betas': [0.005],
                'gammas': [0.0005],
                'perturbated_epsilon': [0.5],
                'hypernetworks_hidden_layers': [[100, 100]],
                'best_model_selection_method': 'val_loss',
                'saving_folder': './Results/'
                f'permuted_mnist_final_grid_experiments/last_model/'
            }

        # Both in the grid search and individual runs
        hyperparams['lr_scheduler'] = False
        hyperparams['number_of_iterations'] = 5000
        hyperparams['number_of_epochs'] = None
        hyperparams['no_of_validation_samples'] = 5000
        hyperparams['target_hidden_layers'] = [1000, 1000]
        hyperparams['target_network'] = 'MLP'
        hyperparams['resnet_number_of_layer_groups'] = None
        hyperparams['resnet_widening_factor'] = None
        hyperparams['optimizer'] = 'adam'
        hyperparams['chunk_size'] = 100
        hyperparams['chunk_emb_size'] = 8
        hyperparams['use_chunks'] = False
        hyperparams['use_batch_norm'] = False
        # Directly related to the MNIST dataset
        hyperparams['padding'] = 2
        hyperparams['shape'] = (28 + 2 * hyperparams['padding'])**2
        hyperparams['number_of_tasks'] = 10
        hyperparams['augmentation'] = False


    elif dataset == 'CIFAR100':
        if grid_search:
            hyperparams = {
                'seed': [1,2,3,4,5],
                'embedding_sizes': [48],
                'betas': [0.01, 0.1, 1.0],
                'gammas': [0.01, 0.1, 1.0],
                'learning_rates': [0.001],
                'batch_sizes': [32],
                'hypernetworks_hidden_layers': [[100]],
                'perturbated_epsilon': [10, 5, 1, 0.5],
                'resnet_number_of_layer_groups': 3,
                'resnet_widening_factor': 2,
                'optimizer': 'adam',
                'use_batch_norm': False,
                'target_network': 'ZenkeNet',
                'use_chunks': False,
                'number_of_epochs': 200,
                'augmentation': True
            }
            if part == 0:
                # Use default grid search hyperparameters
                pass
            elif part == 1:
                hyperparams['betas'] = [0.1]
                hyperparams['gammas'] = [1]
                hyperparams['seed'] = [5, 4]
            elif part == 2:
                hyperparams['betas'] = [1]
                hyperparams['gammas'] = [1]
                hyperparams['seed'] = [5, 4]
            elif part == 6:
                hyperparams['betas'] = [0.01, 0.1, 1, 0.05]
                hyperparams['gammas'] = [0.01]
            elif part == 7:
                hyperparams['target_network'] = 'ZenkeNet'
                hyperparams['resnet_number_of_layer_groups'] = None
                hyperparams['resnet_widening_factor'] = None
                hyperparams['number_of_epochs'] = 200
                hyperparams['use_chunks'] = False
                hyperparams['use_batch_norm'] = False
                hyperparams['seed'] = [4, 5]
                hyperparams['betas'] = [0.01]
                hyperparams['gammas'] = [0.01]
            else:
                raise ValueError(f'Wrong argument: {part}!')
            hyperparams['saving_folder'] = (
                '/shared/results/pkrukowski/HyperIntervalResults/non_forced_intervals/grid_search_relu/'
                f'CIFAR-100_single_seed/'
                f'part_{part}_ResNet_stronger_reg/'
            )

        else:
            # single run experiment
            hyperparams = {
                'seed': [3],
                'embedding_sizes': [48],
                'betas': [0.01],
                'gammas': [0.0001],
                'batch_sizes': [32],
                'learning_rates': [0.001],
                'perturbated_epsilon': [0.5],
                'hypernetworks_hidden_layers': [[100]],
                'use_batch_norm': False,
                'use_chunks': False,
                'resnet_number_of_layer_groups': 3,
                'resnet_widening_factor': 2,
                'number_of_epochs': 200,
                'target_network': 'ResNet',
                'optimizer': 'adam',
                'augmentation': True
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams['embedding_sizes'] = [48]
                hyperparams['hypernetworks_hidden_layers'] = [[100]]
                hyperparams['betas'] = [0.01]
                hyperparams['gammas'] = [0.01]
                hyperparams['target_network'] = 'ZenkeNet'
                hyperparams['resnet_number_of_layer_groups'] = None
                hyperparams['resnet_widening_factor'] = None
            else:
                raise ValueError(f'Wrong argument: {part}!')
            hyperparams['saving_folder'] = (
                './Results/grid_search_relu/'
                f'CIFAR-100_single_seed/'
                f'part_{part}_final_run/'
            )
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 500
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 10
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    elif dataset == 'SplitMNIST':
        if grid_search:
            hyperparams = {
                'learning_rates': [0.001],
                'batch_sizes': [64, 128],
                'betas': [0.01, 0.001],
                'hypernetworks_hidden_layers': [[25, 25], [50, 50]],
                'perturbated_epsilon': [10, 5, 1, 0.5],
                'gammas': [0.01, 0.001],
                # seed is not for optimization but for ensuring multiple results
                'seed': [1, 2, 3, 4, 5],
                'best_model_selection_method': 'val_loss',
                'embedding_sizes': [8, 16, 24],
                'augmentation': True
            }
            if part == 0:
                pass
            elif part == 1:
                hyperparams['embedding_sizes'] = [96]
                hyperparams['hypernetworks_hidden_layers'] = [[50, 50]]
                hyperparams['betas'] = [0.01]
                hyperparams['gammas'] = [0.0001]
            else:
                raise ValueError('Not implemented subset of hyperparameters!')

            hyperparams['saving_folder'] = (
                '/shared/results/pkrukowski/HyperIntervalResults/non_forced_intervals/grid_search_relu/'
                f'split_mnist/augmented/ICLR_models/part_{part}/'
            )

        else:
            # single run experiment
            hyperparams = {
                'seed': [3],
                'embedding_sizes': [128],
                'learning_rates': [0.001],
                'batch_sizes': [1],
                'betas': [0.001],
                'perturbated_epsilon': [0.5],
                'gammas': [0.001],
                'hypernetworks_hidden_layers': [[25, 25]],
                'augmentation': True,
                'best_model_selection_method': 'last_model',
                'saving_folder': './Results/SplitMNIST/norm_test/'
            }
        hyperparams['lr_scheduler'] = False
        hyperparams['target_network'] = 'MLP'
        hyperparams['resnet_number_of_layer_groups'] = None
        hyperparams['resnet_widening_factor'] = None
        hyperparams['optimizer'] = 'adam'
        hyperparams['number_of_iterations'] = 2000
        hyperparams['number_of_epochs'] = None
        hyperparams['no_of_validation_samples'] = 1000
        hyperparams['target_hidden_layers'] = [2, 3]
        hyperparams['shape'] = 28**2
        hyperparams['number_of_tasks'] = 5
        hyperparams['chunk_size'] = 100
        hyperparams['chunk_emb_size'] = 96
        hyperparams['use_chunks'] = False
        hyperparams['use_batch_norm'] = False
        hyperparams['padding'] = None
        # hyperparams['best_model_selection_method'] = 'val_loss'
    
    elif dataset == 'TinyImageNet':
        if grid_search:
            hyperparams = {
                "seed": [1,2,3,4,5],
                "perturbated_epsilon": [10, 5, 1, 0.5],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                'betas': [0.01, 0.1, 1.0],
                'gammas': [0.01, 0.1, 1.0],
                'hypernetworks_hidden_layers': [[100], [10, 10]],
                'resnet_number_of_layer_groups': 3,
                'resnet_widening_factor': 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True
            }

            hyperparams['saving_folder'] = (
                '/shared/results/pkrukowski/HyperIntervalResults/non_forced_intervals/grid_search_relu/'
                f'TinyImageNet/'
                f'part_0_ZenkeNet/'
            )
        else:
            hyperparams = {
               "seed": [1],
                "perturbated_epsilon": [1.0],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                'betas': [0.01],
                'gammas': [0.01],
                'hypernetworks_hidden_layers': [[100]],
                'resnet_number_of_layer_groups': 3,
                'resnet_widening_factor': 2,
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": "./Results/TinyImageNet/best_hyperparams/"
            }
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 250
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 1
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

    else:
        raise ValueError('This dataset is not implemented!')

    # General hyperparameters
    hyperparams['activation_function'] = torch.nn.ReLU()
    hyperparams['norm'] = 1  # L1 norm
    hyperparams['use_bias'] = True
    hyperparams["dataset"] = dataset
    hyperparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hyperparams['kappa'] = 0.5
    os.makedirs(hyperparams['saving_folder'], exist_ok=True)
    return hyperparams


if __name__ == "__main__":
    datasets_folder = './Data'
    #datasets_folder = '/shared/sets/datasets/'
    os.makedirs(datasets_folder, exist_ok=True)
    validation_size = 500
    use_data_augmentation = False
    use_cutout = False

    split_cifar100_list = prepare_split_cifar100_tasks(
        datasets_folder,
        validation_size,
        use_data_augmentation,
        use_cutout
    )
