import torch
import torch.nn.functional as F
import numpy as np
from main import reverse_predictions

from numpy.testing import assert_array_equal


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


def unittest_translate_output_MNIST_classes():
    """
    Unittest of translate_output_MNIST_classes() function.
    """
    labels = [0, 0, 1, 0]
    test_1 = translate_output_MNIST_classes(labels, 3, "split")
    gt_1 = np.array([6, 6, 7, 6])
    assert_array_equal(test_1, gt_1)

    labels = [0, 1, 1, 0]
    test_2 = translate_output_MNIST_classes(labels, 0, "split")
    gt_2 = np.array([0, 1, 1, 0])
    assert_array_equal(test_2, gt_2)

    labels = [0, 5, 7, 0, 8, 9]
    test_3 = translate_output_MNIST_classes(labels, 0, "permuted")
    gt_3 = np.array([0, 5, 7, 0, 8, 9])
    assert_array_equal(test_3, gt_3)

    test_4 = translate_output_MNIST_classes(labels, 5, "permuted")
    gt_4 = np.array([50, 55, 57, 50, 58, 59])
    assert_array_equal(test_4, gt_4)


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



if __name__ == "__main__":
   pass