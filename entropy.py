import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from datetime import datetime

from evaluation import (
    prepare_and_load_weights_for_models,
)

from main import reverse_predictions


def translate_output_CIFAR_classes(labels, setup, task, mode):
    """
    Translate labels of form {0, 1, ..., N-1} to the real labels
    of CIFAR100 dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form {0, 1, ..., N-1}
                 where N is the the number of classes in a single task
       *setup*: (int) defines how many tasks were created in this
                training session
       *task*: (int) number of the currently calculated task
       *mode*: (str) defines if dataset is CIFAR100 or CIFAR10, available values:
            - CIFAR100,
            - CIFAR10
    Returns:
    --------
       A numpy array of the same shape like *labels* but with proper
       class labels
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


def unittest_translate_output_CIFAR_classes():
    """
    Unittest of translate_output_CIFAR_classes() function.
    """
    # 21 tasks
    labels = [i for i in range(40)]
    test_1 = translate_output_CIFAR_classes(labels, 21, 0)
    gt_1 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
            84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
            69, 66, 77, 47, 79, 93, 29, 50, 57, 83]
    assert (test_1 == gt_1).all()
    labels = [i for i in range(3)]
    test_2 = translate_output_CIFAR_classes(labels, 21, 1)
    gt_2 = [17, 81, 41]
    assert (test_2 == gt_2).all()
    test_3 = translate_output_CIFAR_classes(labels, 21, 20)
    gt_3 = [26, 35, 39]
    assert (test_3 == gt_3).all()
    # 11 tasks
    labels = [i for i in range(50)]
    test_4 = translate_output_CIFAR_classes(labels, 11, 0)
    gt_4 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
            84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
            69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
            17, 81, 41, 12, 37, 59, 25, 20, 80, 73]
    assert (test_4 == gt_4).all()
    labels = [i for i in range(5)]
    test_5 = translate_output_CIFAR_classes(labels, 11, 2)
    gt_5 = [82, 53, 9, 31, 75]
    assert (test_5 == gt_5).all()
    # 6 tasks
    labels = [i for i in range(50)]
    test_6 = translate_output_CIFAR_classes(labels, 6, 0)
    assert (test_6 == gt_4).all()
    labels = [i for i in range(10)]
    test_7 = translate_output_CIFAR_classes(labels, 6, 4)
    gt_7 = [40, 30, 23, 85, 2, 95, 56, 48, 71, 64]
    assert (test_7 == gt_7).all()
    # 5 tasks
    labels = [i for i in range(20)]
    test_8 = translate_output_CIFAR_classes(labels, 5, 0)
    gt_8 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            94, 92, 10, 72, 49, 78, 61, 14, 8, 86]
    assert (test_8 == gt_8).all()
    test_9 = translate_output_CIFAR_classes(labels, 5, 3)
    gt_9 = [38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
            60, 19, 70, 90, 89, 43, 5, 42, 65, 76]
    assert (test_9 == gt_9).all()


def translate_output_MNIST_classes(relative_labels, task, mode):
    """
    Translate relative labels of form {0, 1} to the real labels
    of Split MNIST dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form
       *task*: (int) number of the currently calculated task,
               starting from 0
       *mode*: (str) "permuted" or "split", depending on the desired
               dataset
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
    having a hypernetwork with its weights, and a target network with
    its weights, as well as the number of the considered task.

    Arguments:
    ----------
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_weights*: loaded weights for the target network
       *target_network_type*: str representing the target network architecture
       *input_data*: torch.Tensor with input data for the network
       *task*: int representing the considered task; the corresponding
               embedding and batch normalization statistics will be used
       *perturbated_eps*: float representing taken perturbated epsilon
       *full_interval*: bool indicating whether we have a proper interval
                        mechanism or not

    Returns:
    --------
       A list containing torch.Tensor (or tensors) representing lower, 
       middle and upper values from the output classification layer
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

        if target_network_type == "ResNet":
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
    dataset_CL_tasks, no_of_task, dataset, device
):
    """
    Extract test samples dedicated for a selected task
    and change relative output classes into absolute classes.

    Arguments:
    ----------
       *dataset_CL_tasks*: list of objects containing consecutive tasks
       *no_of_task*: (int) represents number of the currently analyzed task
       *dataset*: (str) defines name of the dataset used: 'PermutedMNIST',
                  'SplitMNIST' or 'CIFAR100_FeCAM_setup'
       *device*: (str) defines whether CPU or GPU will be used

    Returns:
    --------
       *X_test*: (torch.Tensor) represents input samples
       *gt_classes*: (Numpy array) represents absolute classes for *X_test*
       *gt_tasks*: (list) represents number of task for corresponding samples
    """
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
    if dataset == "CIFAR100_FeCAM_setup":
        # Currently there is an assumption that only setup with
        # 5 tasks will be used
        gt_classes = translate_output_CIFAR_classes(
            gt_classes, setup=5, task=no_of_task
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
    with corresponding labels (without forward propagation through network)
    and information about task.

    Arguments:
    ----------
       *dataset_CL_tasks* (list of datasets) list of objects storing training
                           and test samples from consecutive tasks
       *number_of_incremental_tasks* (int) the number of consecutive tasks
                                      from which the test sets will be
                                      extracted
       *total_number_of_tasks* (int) the number of all tasks in a given
                               experiment
       *device*: (str) 'cpu' or 'cuda', defines the equipment for computations

    Returns:
    --------
       *X_test* (torch Tensor) contains samples from the test set,
                shape: (number of samples, number of image features [e.g. 3072
                for CIFAR-100])
       *y_test* (Numpy array) contains labels for corresponding samples
                from *X_test* (number of samples, )
       *tasks_test* (Numpy array) contains information about task for
                    corresponding samples from *X_test* (number of samples, )
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
    Get task prediction for consecutive samples based on interval 
    entropy values of the output classification layer of the target network.

    Arguments:
    ----------
       *inferenced_logits_of_all_tasks*: shape: (number of tasks,
                            number of samples, number of output heads)
       *setup*: (int) defines how many tasks were performed in this
                experiment (in total)
       *dataset*: (str) name of the dataset for proper class translation

    Returns:
    --------
       *predicted_tasks*: torch.Tensor with the prediction of tasks for
                          consecutive samples
       *predicted_classes*: torch.Tensor with the prediction of classes for
                            consecutive samples.
       Positions of samples in the two above Tensors are the same.
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

            softmaxed_inferred_task = F.softmax(
                all_task_single_output_sample[no_of_inferred_task, 1, :], dim=-1
            )
            
            if not vanilla_entropy:
                lower_logits = all_task_single_output_sample[no_of_inferred_task, 0, :]
                upper_logits = all_task_single_output_sample[no_of_inferred_task, 2, :]

                factor = 1 /(0.001 + (upper_logits - lower_logits).abs())

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
    Select the target task automatically and calculate accuracy for
    consecutive samples

    Arguments:
    ----------
    *experiment_models*: A dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *hypernetwork_weights*: loaded weights for the hypernetwork
       *target_network*: an instance of MLP or ResNet class
       *target_network_weights*: loaded weights for the target network
       *hyperparameters*: a dictionary with experiment's hyperparameters
       *dataset_CL_tasks*: list of objects containing consecutive tasks

    Returns Pandas Dataframe with results for the selected model.
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
    # The results are varying depending on the batch sizes due to the fact
    # that batch normalization is turned on in ResNet. We selected 2000 as
    # the test set size to ensure that it is derived to the network
    # in one piece.
    
    # alphas = np.linspace(0.01, 1.0, 5)
    alphas = [1.0]
    vanilla_entropy = True

    # Options for *dataset*:
    # 'PermutedMNIST', 'SplitMNIST', 'CIFAR100_FeCAM_setup', 'SubsetImageNet', 'CIFAR10'
    dataset = "SplitMNIST"
    path_to_datasets = "./Data/"

    for alpha in alphas:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Generate timestamp
        path_to_stored_networks = f"./SavedModels/{dataset}/known_task_id/"
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


