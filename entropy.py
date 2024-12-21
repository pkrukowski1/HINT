import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from datetime import datetime

from evaluation import (
    prepare_and_load_weights_for_models,
)

from Utils.handy_functions import reverse_predictions


def translate_output_CIFAR_classes(labels, setup, task, mode):
    """
    Translate labels of the form {0, 1, ..., N-1} to the real labels
    of the CIFAR100 dataset.

    Parameters:
    -----
    labels: Union[np.ndarray, List[int]]
        Contains labels of the form {0, 1, ..., N-1} where N is the number
        of classes in a single task.
    setup: int
        Defines how many tasks were created in this training session.
    task: int
        Number of the currently calculated task.
    mode: str
        Defines if the dataset is CIFAR100 or CIFAR10. Available values:
        - "CIFAR100"
        - "CIFAR10"

    Returns:
    --------
    np.ndarray
        A numpy array of the same shape as `labels` but with proper
        class labels.
    """
    assert setup in [5, 6, 11, 21]
    assert mode in ["CIFAR100", "CIFAR10"]
    # 5 tasks: 20 classes in each task
    # 6 tasks: 50 initial classes + 5 incremental tasks per 10 classes
    # 11 tasks: 50 initial classes + 10 incremental tasks per 5 classes
    # 21 tasks: 40 initial classes + 20 incremental tasks per 3 classes

    if mode == "CIFAR100":
        class_orders = [
            87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
            84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
            69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
            17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
            1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
            38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
            60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
            40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
            98, 13, 99, 7, 34, 55, 54, 26, 35, 39
        ]
        if setup in [6, 11]:
            no_of_initial_cls = 50
        elif setup == 21:
            no_of_initial_cls = 40
        else:
            no_of_initial_cls = 20
        if task == 0:
            currently_used_classes = class_orders[:no_of_initial_cls]
        else:
            if setup == 6:
                no_of_incremental_cls = 10
            elif setup == 11:
                no_of_incremental_cls = 5
            elif setup == 21:
                no_of_incremental_cls = 3
            else:
                no_of_incremental_cls = 20
            currently_used_classes = class_orders[
                (no_of_initial_cls + no_of_incremental_cls * (task - 1)) : (
                    no_of_initial_cls + no_of_incremental_cls * task
                )
            ]
    else:
        total_no_of_classes = 10
        no_of_classes_per_task = 2

        class_orders = [i for i in range(total_no_of_classes)]
        currently_used_classes = class_orders[
            (no_of_classes_per_task * task) : (no_of_classes_per_task * (task + 1))
        ]

    y_translated = np.array(
        [currently_used_classes[i] for i in labels]
    )
    return y_translated


def translate_output_MNIST_classes(relative_labels, task, mode):
    """
    Translate relative labels of the form {0, 1} to the real labels
    of the Split MNIST dataset.

    Parameters:
    -----------
    relative_labels: Union[np.ndarray, List[int]]
        Contains labels of the form {0, 1} where 0 represents the first class
        and 1 represents the second class.
    task: int
        Number of the currently calculated task (starting from 0).
    mode: str
        Defines if the dataset is "permuted" or "split", depending on the desired
        dataset.

    Returns:
    --------
    np.ndarray
        A numpy array of the same shape as `relative_labels` but with proper
        class labels.
    """
    assert mode in ["permuted", "split"]

    if mode == "permuted":
        total_no_of_classes = 100
        no_of_classes_per_task = 10
        # Even if the classifier indicates '0' but from the wrong task
        # it has to get a penalty. Therefore, in Permuted MNIST there
        # are 100 unique classes.
    elif mode == "split":
        total_no_of_classes = 10
        no_of_classes_per_task = 2

    class_orders = [i for i in range(total_no_of_classes)]
    currently_used_classes = class_orders[
        (no_of_classes_per_task * task) : (no_of_classes_per_task * (task + 1))
    ]

    y_translated = np.array(
        [currently_used_classes[i] for i in relative_labels]
    )
    return y_translated

def get_target_network_representation(
    hypernetwork,
    hypernetwork_weights,
    target_network,
    target_network_type,
    input_data,
    task,
    perturbated_eps,
    full_interval
):
    """
    Calculate the output classification layer of the target network,
    using a hypernetwork with its weights and a target network with
    its weights, along with the number of the considered task.

    Parameters:
    -----------
    hypernetwork: HMLP
        An instance of the hypernetwork class.
    hypernetwork_weights: torch.Tensor
        Loaded weights for the hypernetwork.
    target_network: MLP or ResNet
        An instance of the target network class.
    target_network_type: str
        Represents the target network architecture ("MLP" or "ResNet").
    input_data: torch.Tensor
        Input data for the network.
    task: int
        The considered task; the corresponding embedding and batch normalization
        statistics will be used (if applicable)
    perturbated_eps: float
        Represents the taken perturbated epsilon.
    full_interval: bool
        Indicates whether a proper interval mechanism is used or not.

    Returns:
    --------
    torch.Tensor or list of torch.Tensor
        A tensor (or list of tensors) representing lower, middle, and upper
        values from the output classification layer.
    """
    hypernetwork.eval()
    target_network.eval()

    with torch.no_grad():

        
        (
            lower_target_weights,
            middle_target_weights,
            upper_target_weights,
            _
        ) = hypernetwork.forward(
            cond_id=task, 
            weights=hypernetwork_weights,
            perturbated_eps=perturbated_eps,
            return_extended_output=True
        )

        if target_network_type in ["ResNet", "AlexNet"]:
                condition = task
        else:
            condition = None
        
        if full_interval:

            # Lower, middle and upper logits!
            logits = target_network.forward(
                                        input_data,
                                        lower_weights=lower_target_weights,
                                        middle_weights=middle_target_weights,
                                        upper_weights=upper_target_weights,
                                        condition=condition
                                    )
            logits = logits.rename(None)
            

        else:
            logits = torch.stack(reverse_predictions(
                                    target_network,
                                    input_data,
                                    lower_target_weights,
                                    middle_target_weights,
                                    upper_target_weights,
                                    condition
                                ), dim=1)
            
        
        
        logits = logits.detach().cpu()
    
    return logits

def extract_test_set_from_single_task(
    dataset_CL_tasks, no_of_task, dataset, device, mode="CIFAR100"
):
    """
    Extract test samples dedicated for a selected task and change relative
    output classes into absolute classes.

    Parameters:
    -----------
    dataset_CL_tasks: List[object]
        List of objects containing consecutive tasks.
    no_of_task: int
        Represents the number of the currently analyzed task.
    dataset: str
        Defines the name of the dataset used: 'PermutedMNIST', 'SplitMNIST',
        or 'CIFAR100_FeCAM_setup'.
    device: str
        Defines whether CPU or GPU will be used.
    mode: str
        Defines number of classes in CIFAR, e.g. `CIFAR100` or `CIFAR10`

    Returns:
    --------
    Tuple[torch.Tensor, np.ndarray, List[int]]
        A tuple containing:
        - X_test: torch.Tensor representing input samples.
        - gt_classes: Numpy array representing absolute classes for X_test.
        - gt_tasks: List representing the number of the task for corresponding samples.
    """

    assert mode in ["CIFAR100", "CIFAR10"]

    tested_task = dataset_CL_tasks[no_of_task]
    input_data = tested_task.get_test_inputs()
    output_data = tested_task.get_test_outputs()
    X_test = tested_task.input_to_torch_tensor(
        input_data, device, mode="inference"
    )
    test_output = tested_task.output_to_torch_tensor(
        output_data, device, mode="inference"
    )
    gt_classes = test_output.max(dim=1)[1]
    if dataset in ["CIFAR100_FeCAM_setup", "CIFAR10"]:
        # Currently there is an assumption that only setup with
        # 5 tasks will be used for CIFAR100_FeCAM_setup
        mode = "CIFAR100" if dataset == "CIFAR100_FeCAM_setup" else "CIFAR10"
        gt_classes = translate_output_CIFAR_classes(
            gt_classes, setup=5, task=no_of_task, mode=mode
        )
    elif dataset in ["PermutedMNIST", "SplitMNIST"]:
        mode = "permuted" if dataset == "PermutedMNIST" else "split"
        gt_classes = translate_output_MNIST_classes(
            gt_classes, task=no_of_task, mode=mode
        )
    elif dataset == "SubsetImageNet":
        raise NotImplementedError
    else:
        raise ValueError("Wrong name of the dataset!")
    gt_tasks = [no_of_task for _ in range(output_data.shape[0])]
    return X_test, gt_classes, gt_tasks



def extract_test_set_from_all_tasks(
    dataset_CL_tasks, number_of_incremental_tasks, total_number_of_tasks, device
):
    """
    Create a test set containing samples from all the considered tasks
    with corresponding labels (without forward propagation through the network)
    and information about the task.

    Parameters:
    -----------
    dataset_CL_tasks: List[object]
        List of objects storing training and test samples from consecutive tasks.
    number_of_incremental_tasks: int
        The number of consecutive tasks from which the test sets will be extracted.
    total_number_of_tasks: int
        The total number of all tasks in a given experiment.
    device: str
        Defines whether CPU or GPU will be used.

    Returns:
    --------
    Tuple[torch.Tensor, np.ndarray, np.ndarray]
        A tuple containing:
        - X_test: torch.Tensor containing samples from the test set
          (shape: number of samples, number of image features [e.g., 3072 for CIFAR-100]).
        - y_test: Numpy array containing labels for corresponding samples from X_test (shape: number of samples).
        - tasks_test: Numpy array containing information about the task for corresponding samples from X_test (shape: number of samples).
    """

    test_input_data, test_output_data, test_ID_tasks = [], [], []
    for t in range(number_of_incremental_tasks):
        tested_task = dataset_CL_tasks[t]
        input_test_data = tested_task.get_test_inputs()
        output_test_data = tested_task.get_test_outputs()
        test_input = tested_task.input_to_torch_tensor(
            input_test_data, device, mode="inference"
        )
        test_output = tested_task.output_to_torch_tensor(
            output_test_data, device, mode="inference"
        )
        gt_classes = test_output.max(dim=1)[1].cpu().detach().numpy()
        gt_classes = translate_output_CIFAR_classes(
            gt_classes, total_number_of_tasks, t
        )
        test_input_data.append(test_input)
        test_output_data.append(gt_classes)
        current_task_gt = np.zeros_like(gt_classes) + t
        test_ID_tasks.append(current_task_gt)
    X_test = torch.cat(test_input_data)
    y_test, tasks_test = np.concatenate(test_output_data), np.concatenate(
        test_ID_tasks
    )
    assert X_test.shape[0] == y_test.shape[0] == tasks_test.shape[0]
    return X_test, y_test, tasks_test

def get_task_and_class_prediction_based_on_logits(
    inferenced_logits_of_all_tasks, setup, dataset, vanilla_entropy = False
):
    """
    Get task predictions for consecutive samples based on interval entropy values
    of the output classification layer of the target network.

    Parameters:
    -----------
    inferenced_logits_of_all_tasks: torch.Tensor
        Shape: (number of tasks, number of samples, 3, number of output heads).
    setup: int
        Defines how many tasks were performed in this experiment (in total).
    dataset: str
        Name of the dataset for proper class translation.
    vanilla_entropy: bool, optional
        Indicates whether vanilla entropy calculation should be used (default: False).

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - predicted_tasks: torch.Tensor with the prediction of tasks for consecutive samples.
        - predicted_classes: torch.Tensor with the prediction of classes for consecutive samples.
          Positions of samples in the two tensors are the same.
    """
    predicted_classes, predicted_tasks = [], []
    number_of_samples = inferenced_logits_of_all_tasks.shape[1]

    for no_of_sample in range(number_of_samples):
        task_entropies = torch.zeros((inferenced_logits_of_all_tasks.shape[0]))

        all_task_single_output_sample = inferenced_logits_of_all_tasks[
            :, no_of_sample, :, :
        ]

        # Calculate entropy based on results from all tasks
        for no_of_inferred_task in range(task_entropies.shape[0]):

            lower_logits = all_task_single_output_sample[no_of_inferred_task, 0, :]
            upper_logits = all_task_single_output_sample[no_of_inferred_task, 2, :]
            
            softmaxed_inferred_task = F.softmax((lower_logits + upper_logits)/2.0, dim=-1)


            if not vanilla_entropy:
                factor = 1 /(upper_logits - lower_logits + 1e-8).abs()

                assert not torch.isnan(factor).any()
            else:
                factor = 1.0
            
            task_entropies[no_of_inferred_task] = -1 * torch.sum(factor * \
                softmaxed_inferred_task * torch.log(softmaxed_inferred_task), dim=-1
            )
        
        selected_task_id = torch.argmin(task_entropies)
        predicted_tasks.append(selected_task_id.item())

        # We evaluate performance of classification task on middle
        # logits only 
        target_output = all_task_single_output_sample[selected_task_id.item(), 1, :]

        output_relative_class = target_output.argmax().item()

        if dataset in ["CIFAR100_FeCAM_setup", "CIFAR10"]:
            mode = "CIFAR100" if dataset == "CIFAR100_FeCAM_setup" else "CIFAR10"
            output_absolute_class = translate_output_CIFAR_classes(
                [output_relative_class], setup, selected_task_id.item(), mode=mode
            )
        elif dataset in ["PermutedMNIST", "SplitMNIST"]:
            mode = "permuted" if dataset == "PermutedMNIST" else "split"
            output_absolute_class = translate_output_MNIST_classes(
                [output_relative_class], selected_task_id.item(), mode=mode
            )
        else:
            raise ValueError("Wrong name of the dataset!")
        predicted_classes.append(output_absolute_class)
    predicted_tasks = torch.tensor(predicted_tasks, dtype=torch.int32)
    predicted_classes = torch.tensor(predicted_classes, dtype=torch.int32)
    return predicted_tasks, predicted_classes


def calculate_entropy_and_predict_classes_separately(experiment_models):
    """
    Select the target task automatically and calculate accuracy for consecutive samples.

    Parameters:
    -----------
    experiment_models: dict
        A dictionary with the following keys:
        - "hypernetwork": An instance of the HMLP class.
        - "hypernetwork_weights": Loaded weights for the hypernetwork.
        - "target_network": An instance of the target network
        - "target_network_weights": Loaded weights for the target network.
        - "hyperparameters": A dictionary with experiment's hyperparameters.
        - "dataset_CL_tasks": List of objects containing consecutive tasks.

    Returns:
    --------
    pd.DataFrame
        A Pandas DataFrame with results for the selected model.
    """

    hypernetwork = experiment_models["hypernetwork"]
    hypernetwork_weights = experiment_models["hypernetwork_weights"]
    target_network = experiment_models["target_network"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    dataset_name = experiment_models["hyperparameters"]["dataset"]
    target_network_type = hyperparameters["target_network"]
    saving_folder = hyperparameters["saving_folder"]
    alpha = hyperparameters["alpha"]
    full_interval = hyperparameters["full_interval"]
    vanilla_entropy = experiment_models["vanilla_entropy"]

    hypernetwork.eval()
    target_network.eval()

    results = []
    for task in range(hyperparameters["number_of_tasks"]):

        X_test, y_test, gt_tasks = extract_test_set_from_single_task(
            dataset_CL_tasks, task, dataset_name, hyperparameters["device"]
        )

        with torch.no_grad():
            logits_outputs_for_different_tasks = []
            for inferenced_task in range(hyperparameters["number_of_tasks"]):

                # Try to predict task for all samples from "task"
                logits = get_target_network_representation(
                    hypernetwork,
                    hypernetwork_weights,
                    target_network,
                    target_network_type,
                    X_test,
                    inferenced_task,
                    alpha,
                    full_interval
                )

                logits_outputs_for_different_tasks.append(logits)

            all_inferenced_tasks = torch.stack(
                logits_outputs_for_different_tasks
            )
            # Sizes of consecutive dimensions represent:
            # number of tasks x number of samples x 3 x number of output heads
        (
            predicted_tasks,
            predicted_classes,
        ) = get_task_and_class_prediction_based_on_logits(
            all_inferenced_tasks,
            hyperparameters["number_of_tasks"],
            dataset_name,
            vanilla_entropy=vanilla_entropy
        )
        predicted_classes = predicted_classes.flatten().numpy()
        task_prediction_accuracy = (
            torch.sum(predicted_tasks == task).float()
            * 100.0
            / predicted_tasks.shape[0]
        ).item()
        print(f"task prediction accuracy: {task_prediction_accuracy}")
        sample_prediction_accuracy = (
            np.sum(predicted_classes == y_test) * 100.0 / y_test.shape[0]
        ).item()
        print(f"sample prediction accuracy: {sample_prediction_accuracy}")
        results.append(
            [task, task_prediction_accuracy, sample_prediction_accuracy]
        )
    results = pd.DataFrame(
        results, columns=["task", "task_prediction_acc", "class_prediction_acc"]
    )
    results.to_csv(
        f"{saving_folder}entropy_statistics_{number_of_model}.csv", sep=";"
    )
    return results


if __name__ == "__main__":
    
    # alphas = np.linspace(0.01, 0.5, 5)
    alphas = [0.1]
    vanilla_entropy = False

    # Options for *dataset*:
    # 'PermutedMNIST', 'SplitMNIST', 'CIFAR100_FeCAM_setup', 'CIFAR10'
    dataset = "SplitMNIST"
    path_to_datasets = "./Data/"

    for alpha in alphas:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Generate timestamp
        path_to_stored_networks = "./SavedModels/SplitMNIST/known_task_id/"
        path_to_save = f"./Results/{dataset}/{timestamp}/"
        os.makedirs(path_to_save, exist_ok=True)

        results_summary = []
        numbers_of_models = [i for i in range(5)]
        seeds = [i + 1 for i in range(5)]

        dict_to_save = {}

        for number_of_model, seed in zip(numbers_of_models, seeds):
            print(f"Calculations for model no: {number_of_model}")
            experiment_models = prepare_and_load_weights_for_models(
                path_to_stored_networks,
                path_to_datasets,
                number_of_model,
                dataset,
                seed=seed,
            )

            experiment_models["hyperparameters"]["saving_folder"] = path_to_save
            experiment_models["hyperparameters"]["alpha"] = alpha
            experiment_models["vanilla_entropy"] = vanilla_entropy

            results = calculate_entropy_and_predict_classes_separately(
                experiment_models
            )
            results_summary.append(results)
            
        data_statistics = []
        for summary in results_summary:
            data_statistics.append(
                [
                    list(summary["task_prediction_acc"].values),
                    list(summary["class_prediction_acc"].values),
                    np.mean(summary["task_prediction_acc"].values),
                    np.std(summary["task_prediction_acc"].values),
                    np.mean(summary["class_prediction_acc"].values),
                    np.std(summary["class_prediction_acc"].values),
                ]
            )
        column_names = [
            "task_prediction_accuracy",
            "class_prediction_accuracy",
            "mean_task_prediction_accuracy",
            "std_dev_task_prediction_accuracy",
            "mean_class_prediction_accuracy",
            "std_dev_class_prediction_accuracy",
        ]
        table_to_save = data_statistics
        dataframe = pd.DataFrame(table_to_save, columns=column_names)
        dataframe.to_csv(
            f"{path_to_save}entropy_mean_results",
            sep=";",
        )

        dict_to_save["alpha"] = [experiment_models["hyperparameters"]["alpha"]]
        dict_to_save["final_mean"] = np.mean(dataframe["mean_class_prediction_accuracy"])
        dict_to_save["final_stdev"] = np.std(dataframe["mean_class_prediction_accuracy"])
        dataframe = pd.DataFrame.from_dict(dict_to_save)
        dataframe.to_csv(
            f"{path_to_save}hyperparameters",
            sep=";",
        )


