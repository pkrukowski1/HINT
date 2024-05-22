"""
This file implements functions to deal with datasets and tasks in continual learning for Task Incremental Learning scenario
"""

import os
import torch

def set_hyperparameters(dataset,
                        grid_search=False):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Parameters:
    -----------
    dataset: str
        Dataset name ("PermutedMNIST", "SplitMNIST", or "CIFAR100").
    grid_search: bool, optional
        Defines whether hyperparameter optimization should be performed
        (default: False).

    Returns:
    --------
    dict
        A dictionary with necessary hyperparameters.
    """

    if dataset == "PermutedMNIST":
        if grid_search:
            hyperparams = {
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[100, 100]],
                "perturbated_epsilon": [0.5],
                "best_model_selection_method": "val_loss",
                "dropout_rate": [-1],
                # not for optimization
                "seed": [1, 2, 3, 4, 5]
            }

            # hyperparams = {
            #     "embedding_sizes": [24],
            #     "learning_rates": [0.001, 0.01],
            #     "batch_sizes": [128],
            #     "betas": [0.01, 0.005, 0.1],
            #     "hypernetworks_hidden_layers": [[100, 100], [200, 200]],
            #     "perturbated_epsilon": [0.0, 0.5, 1.0, 5.0],
            #     "best_model_selection_method": "val_loss",
            #     "dropout_rate": [-1],
            #     # not for optimization
            #     "seed": [1]
            # }
            
            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f'permuted_mnist_final_grid_experiments/{hyperparams["best_model_selection_method"]}/'
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [3],
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01],
                "perturbated_epsilon": [0.5],
                "hypernetworks_hidden_layers": [[100, 100]],
                "dropout_rate": [-1],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/"
                f"permuted_mnist_final_grid_experiments/last_model/"
            }

        # Both in the grid search and individual runs
        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = 5000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 500
        hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        # Directly related to the MNIST dataset
        hyperparams["padding"] = 2
        hyperparams["shape"] = (28 + 2 * hyperparams["padding"])**2
        hyperparams["number_of_tasks"] = 10
        hyperparams["augmentation"] = False

        # Full-interval model or simpler one
        hyperparams["full_interval"] = True

    elif dataset == "CIFAR100":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [48],
                "betas": [0.01],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[100]],
                "perturbated_epsilon": [5.0],
                "dropout_rate": [-1],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True
            }
        
            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f"CIFAR-100_single_seed/"
                f"ResNet/"
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [48],
                "betas": [0.01],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "perturbated_epsilon": [1.0],
                "hypernetworks_hidden_layers": [[100]],
                "dropout_rate": [-1],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True
            }
           
            hyperparams["saving_folder"] = (
                "./Results/grid_search_relu/"
                f"CIFAR-100_single_seed/"
                f"final_run/"
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
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False

    elif dataset == "SplitMNIST":
        if grid_search:
            # hyperparams = {
            #     "learning_rates": [0.001],
            #     "batch_sizes": [128],
            #     "betas": [0.01, 0.001, 0.1],
            #     "hypernetworks_hidden_layers": [[75, 75], [100, 100]],
            #     "dropout_rate": [-1],
            #     "perturbated_epsilon": [0.0, 0.5, 1.0, 5.0],
            #     # seed is not for optimization but for ensuring multiple results
            #     "seed": [1],
            #     "best_model_selection_method": "val_loss",
            #     "embedding_sizes": [24, 72, 96, 128],
            #     "augmentation": True
            # }
            hyperparams = {
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[75, 75]],
                "dropout_rate": [-1],
                "perturbated_epsilon": [1.0],
                # seed is not for optimization but for ensuring multiple results
                "seed": [1, 2, 3, 4, 5],
                "best_model_selection_method": "val_loss",
                "embedding_sizes": [72],
                "augmentation": True
            }


            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f"split_mnist/augmented/"
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [72],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01],
                "perturbated_epsilon": [1.0],
                "dropout_rate": [-1],
                "hypernetworks_hidden_layers": [[75, 75]],
                "augmentation": True,
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/SplitMNIST/"
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 28**2
        hyperparams["number_of_tasks"] = 5
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = True
    
    elif dataset == "TinyImageNet":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "perturbated_epsilon": [5.0, 1.0, 0.0],
                "embedding_sizes": [96, 48, 126],
                "learning_rates": [0.0001],
                "batch_sizes": [16],
                "dropout_rate": [-1],
                "betas": [0.01, 0.1, 1.0],
                "hypernetworks_hidden_layers": [[100, 100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True
            }

            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f"TinyImageNet/"
                f"ResNet/"
            )
        else:
            hyperparams = {
               "seed": [1],
                "perturbated_epsilon": [1.0],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "dropout_rate": [-1, 0.25, 0.5],
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
        hyperparams["number_of_tasks"] = 40
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False
    
    elif dataset == "SubsetImageNet":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "perturbated_epsilon": [5.0, 1.0, 0.5],
                "embedding_sizes": [48, 96],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "dropout_rate": [-1],
                "betas": [0.01, 0.1],
                "hypernetworks_hidden_layers": [[200]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True
            }

            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f"SubsetImageNet/"
                f"ResNet/"
            )
        else:
            hyperparams = {
               "seed": [1],
                "perturbated_epsilon": [1.0],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [256],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "dropout_rate": [-1],
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 2,
                "augmentation": True,
                "saving_folder": "./Results/SubsetImageNet/best_hyperparams/"
            }
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        hyperparams["no_of_validation_samples"] = 2000
        hyperparams["number_of_tasks"] = 5
        
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False
    
    elif dataset == "CIFAR100_FeCAM_setup":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [48, 64],
                "betas": [0.01, 0.1, 1.0],
                "learning_rates": [0.0001, 0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[200], [100]],
                "perturbated_epsilon": [10, 15, 20],
                "dropout_rate": [-1, 0.2],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True,
                "best_model_selection_method": "val_loss"
            }

            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f'CIFAR100_FeCAM_setup/{hyperparams["best_model_selection_method"]}/'
            )
        else:
            # Single experiment
            hyperparams = {
               "seed": [1],
                "embedding_sizes": [48],
                "betas": [0.01],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[200]],
                "perturbated_epsilon": [1.0],
                "dropout_rate": [-1],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True
            }
            # FeCAM considered three incremental scenarios: with 6, 11 and 21 tasks
            # ResNet - parts 0, 1 and 2
            # ZenkeNet - parts 3, 4 and 5
            # Also, one scenario with equal number of classes: ResNet - part 6
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM/"

        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        hyperparams["no_of_validation_samples"] = 2000
        hyperparams["number_of_tasks"] = 5

        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False

    elif dataset == "CIFAR10":
        if grid_search:
            hyperparams = {
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01, 0.001, 0.1],
                "hypernetworks_hidden_layers": [[100, 100], [50, 50], [25, 25]],
                "dropout_rate": [-1],
                "use_batch_norm": False,
                "target_network": "ResNet",
                "perturbated_epsilon": [0.5, 1.0],
                # seed is not for optimization but for ensuring multiple results
                "seed": [1],
                "best_model_selection_method": "val_loss",
                "embedding_sizes": [24, 48, 72, 96],
                "augmentation": False
            }


            hyperparams["saving_folder"] = (
                "/shared/results/HyperIntervalResults/intersections_non_forced/grid_search_relu/"
                f"CIFAR10"
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [72],
                "learning_rates": [0.001],
                "use_batch_norm": False,
                "target_network": "ResNet",
                "batch_sizes": [128],
                "betas": [0.01],
                "perturbated_epsilon": [1.0],
                "dropout_rate": [-1],
                "hypernetworks_hidden_layers": [[75, 75]],
                "augmentation": False,
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/CIFAR10/"
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 32
        hyperparams["number_of_tasks"] = 5
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = True

    else:
        raise ValueError("This dataset is not implemented!")

    # General hyperparameters
    hyperparams["activation_function"] = torch.nn.ReLU()
    hyperparams["use_bias"] = True
    hyperparams["dataset"] = dataset
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["kappa"] = 0.5
    os.makedirs(hyperparams["saving_folder"], exist_ok=True)
    return hyperparams


if __name__ == "__main__":
    pass