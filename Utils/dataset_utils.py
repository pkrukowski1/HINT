import numpy as np
from hypnettorch.data.special import permuted_mnist
from hypnettorch.data.special.split_cifar import SplitCIFAR100Data, SplitCIFAR10Data
from DatasetHandlers.split_mnist import get_split_mnist_handlers
from DatasetHandlers.subset_image_net import SubsetImageNet
from DatasetHandlers.tiny_image_net import TinyImageNet
from DatasetHandlers.cifar100_FeCAM import SplitCIFAR100Data_FeCAM
from DatasetHandlers.split_cub200 import SplitCUB200Data

from hypnettorch.data.special.regression1d_data import ToyRegression
from DatasetHandlers.gaussian_data import get_gmm_tasks

def generate_random_permutations(shape_of_data_instance,
                                 number_of_permutations):
    """
    Prepare a list of random permutations of the selected shape for continual
    learning tasks.

    Parameters:
    ----------
    shape_of_data_instance: int
        A number defining the shape of the dataset.
    number_of_permutations: int
        The number of permutations to be prepared; it corresponds to the total
        number of tasks.

    Returns:
    --------
    list_of_permutations: List[np.ndarray]
        A list of random permutations, each represented as a NumPy array.
    """
    list_of_permutations = []
    for _ in range(number_of_permutations):
        list_of_permutations.append(np.random.permutation(
            shape_of_data_instance))
    return list_of_permutations


def prepare_split_cifar10_tasks(datasets_folder,
                                 validation_size,
                                 use_augmentation,
                                 use_cutout=False,
                                 use_one_hot=True):
    """
    Prepare a list of 5 tasks, each with 2 classes. The i-th task, where i
    is in {0, 1, ..., 4}, will store samples from classes {2*i, 2*i + 1}.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where CIFAR-10 is stored or will be downloaded.
    validation_size: int
        The number of validation samples.
    use_augmentation: bool
        Potentially applies a data augmentation method from hypnettorch.
    use_cutout: bool, optional
        If True, applies the "apply_cutout" option from "torch_input_transforms".
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    handlers: List[SplitCIFAR10Data]
        A list of SplitCIFAR10Data instances, each representing a task.
    """
    handlers = []
    for i in range(0, 10, 2):
        handlers.append(SplitCIFAR10Data(
            datasets_folder,
            use_one_hot=use_one_hot,
            validation_size=validation_size,
            use_data_augmentation=use_augmentation,
            use_cutout=use_cutout,
            labels=range(i, i + 2)
        ))
    return handlers

def prepare_split_cifar100_tasks(datasets_folder,
                                 validation_size,
                                 use_augmentation,
                                 use_cutout=False,
                                 use_one_hot=True):
    """
    Prepare a list of 10 tasks, each with 10 classes. The i-th task, where i
    is in {0, 1, ..., 9}, will store samples from classes {10*i, 10*i + 1, ..., 10*i + 9}.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where CIFAR-100 is stored or will be downloaded.
    validation_size: int
        The number of validation samples.
    use_augmentation: bool
        Potentially applies a data augmentation method from hypnettorch.
    use_cutout: bool, optional
        If True, applies the "apply_cutout" option from "torch_input_transforms".
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    handlers: List[SplitCIFAR100Data]
        A list of SplitCIFAR100Data instances, each representing a task.
    """
    handlers = []
    for i in range(0, 100, 10):
        handlers.append(SplitCIFAR100Data(
            datasets_folder,
            use_one_hot=use_one_hot,
            validation_size=validation_size,
            use_data_augmentation=use_augmentation,
            use_cutout=use_cutout,
            labels=range(i, i + 10)
        ))
    return handlers


def prepare_CUB200_tasks(datasets_folder,
                        validation_size_per_class,
                        number_of_tasks=5,
                        use_one_hot=True):
    """
    Prepare a list of 'number_of_tasks' sequential tasks, each with equally distributed number of classes. 
    The i-th task, where i is in {0, 1, ..., 'number_of_tasks'-1}, will store samples from classes
    {K*i, K*(i+1)-1}, K := 200 // number_of_tasks.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where CUB-200 is stored or will be downloaded.
    validation_size_per_class: int
        The number of validation samples per class.
    number_of_tasks: int
        The number of sequential tasks.
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    handlers: List[SplitCUB200Data]
        A list of SplitCUB200Data instances, each representing a task.
    """
    handlers = []
    no_classes = 200 // number_of_tasks
    for i in range(0, 200, no_classes):
        handlers.append(SplitCUB200Data(
            datasets_folder,
            use_one_hot=use_one_hot,
            validation_size_per_class=validation_size_per_class,
            labels=range(i, i + no_classes)
        ))
    return handlers

def prepare_split_cifar100_tasks_aka_FeCAM(
    datasets_folder,
    number_of_tasks,
    no_of_validation_samples_per_class,
    use_augmentation,
    use_cutout=False,
    use_one_hot=True
):
    """
    Prepare a list of incremental tasks with varying numbers of classes per task.
    The first task contains a higher number of classes (50 or 40), and subsequent
    tasks have fewer classes (20, 10, or 5). The total number of tasks depends on
    the specified configuration (6, 11, or 21).

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where CIFAR-100 is stored or will be downloaded.
    number_of_tasks: int
        Defines how many continual learning tasks will be created. Possible options:
        6, 11, or 21.
    no_of_validation_samples_per_class: int
        The number of validation samples in a single class.
    use_augmentation: bool
        Potentially applies a data augmentation method from hypnettorch.
    use_cutout: bool, optional
        If True, applies the "apply_cutout" option from "torch_input_transforms".
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    tasks: List[IncrementalTaskData]
        A list of IncrementalTaskData instances, each representing an incremental task.
    """
    # FeCAM considered four scenarios: 5, 10 and 20 incremental tasks
    # and 5 tasks with the equal number of classes
    assert number_of_tasks in [5, 6, 11, 20, 21]
    # The order of image classes in the case of FeCAM was not 0-10, 11-20, etc.,
    # but it was chosen randomly by the authors, and was at follows:
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
    # Incremental tasks from Table I, FeCAM
    if number_of_tasks == 6:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([10 for i in range(5)])
    elif number_of_tasks == 11:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([5 for i in range(10)])
    elif number_of_tasks == 21:
        numbers_of_classes_per_tasks = [40]
        numbers_of_classes_per_tasks.extend([3 for i in range(20)])
    # Tasks with the equal number of elements, Table V, FeCAM
    elif number_of_tasks == 5:
        numbers_of_classes_per_tasks = [20 for i in range(5)]
    elif number_of_tasks == 20:
        numbers_of_classes_per_tasks = [5 for _ in range(20)]

    handlers = []
    for i in range(len(numbers_of_classes_per_tasks)):
        current_number_of_tasks = numbers_of_classes_per_tasks[i]
        validation_size = (
            no_of_validation_samples_per_class * current_number_of_tasks
        )
        handlers.append(
            SplitCIFAR100Data_FeCAM(
                datasets_folder,
                use_one_hot=use_one_hot,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=class_orders[
                    (i * current_number_of_tasks) : (
                        (i + 1) * current_number_of_tasks
                    )
                ],
            )
        )
    return handlers


def prepare_subset_imagenet_tasks(
    datasets_folder: str = "./",
    no_of_validation_samples_per_class: int = 50, 
    setting: int = 4,
    use_augmentation = False,
    number_of_tasks = 5,
    batch_size = 16,
    use_one_hot=True
    ):
    """
    Prepare a list of tasks related to the SubsetImageNet dataset according
    to the WSN (Wide-Scale Nearest Neighbor) setup.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where Subset ImageNet is stored or will be downloaded.
    no_of_validation_samples_per_class: int, optional
        The number of validation samples in each task. By default, it is set to 250,
        similar to the WSN setup.
    setting: int, optional
        Defines the number and type of continual learning tasks.
    use_augmentation: bool, optional
    number_of_tasks: int, optional
        The total number of tasks to be created.
    batch_size: int, optional
        Batch size for training.
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    handlers: List[SubsetImageNet]
        A list of SubsetImageNet objects representing the tasks.
    """

    if setting != 4:
        raise ValueError("Only 5 incremental tasks are supported right now!")
    

    handlers = []
    for i in range(number_of_tasks):

        handlers.append(
            SubsetImageNet(
                path=datasets_folder,
                validation_size=no_of_validation_samples_per_class,
                use_one_hot=use_one_hot,
                use_data_augmentation=use_augmentation,
                task_id = i,
                setting = setting,
                batch_size=batch_size
            )
        )

    return handlers


def prepare_permuted_mnist_tasks(datasets_folder,
                                 input_shape,
                                 number_of_tasks,
                                 padding,
                                 validation_size,
                                 use_one_hot=True):
    """
    Prepare a list of tasks related to the PermutedMNIST dataset.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where the MNIST dataset is stored or will be downloaded.
    input_shape: int
        A number defining the shape of the dataset.
    number_of_tasks: int
        The total number of tasks to be created.
    padding: int
        Padding value for the PermutedMNIST dataset.
    validation_size: int
        The number of validation samples.
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.


    Returns:
    --------
    tasks: List[PermutedMNIST]
        A list of PermutedMNIST objects representing the tasks.
    """
    permutations = generate_random_permutations(
        input_shape,
        number_of_tasks
    )
    return permuted_mnist.PermutedMNISTList(
        permutations,
        datasets_folder,
        use_one_hot=use_one_hot,
        padding=padding,
        validation_size=validation_size
    )


def prepare_split_mnist_tasks(datasets_folder,
                              validation_size,
                              use_augmentation,
                              number_of_tasks=5,
                              use_one_hot=True):
    """
    Prepare a list of tasks related to the SplitMNIST dataset. By default,
    it creates 5 tasks containing consecutive pairs of classes:
    [0, 1], [2, 3], [4, 5], [6, 7], and [8, 9].

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where the MNIST dataset is stored or will be downloaded.
    validation_size: int
        The number of validation samples.
    use_augmentation: bool
        Defines whether dataset augmentation will be applied.
    number_of_tasks: int, optional
        A number defining the total number of learning tasks. By default, it is 5.
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    tasks: List[SplitMNIST]
        A list of SplitMNIST objects representing the tasks.
    """
    return get_split_mnist_handlers(
        datasets_folder,
        use_one_hot=use_one_hot,
        validation_size=validation_size,
        num_classes_per_task=2,
        num_tasks=number_of_tasks,
        use_torch_augmentation=use_augmentation
    )

def prepare_tinyimagenet_tasks(
    datasets_folder: str = "./",
    seed: int = 1993,
    validation_size: int = 100, 
    number_of_tasks: int = 40,
    use_one_hot: bool = True
    ):
    """
    Prepare a list of tasks related to the TinyImageNet dataset according
    to the WSN setup.

    Parameters:
    ----------
    datasets_folder: str
        Defines the path where TinyImageNet is stored or will be downloaded.
    seed: int
        Necessary for preparing random permutations of the order of classes
        in consecutive tasks.
    validation_size: int, optional
        The number of validation samples in each task. By default, it is 250,
        similar to the WSN setup.
    number_of_tasks: int, optional
        Defines the number of continual learning tasks. By default, it is 40.
    use_one_hot: bool, Optional
        If True, then one-hot encoding is applied.

    Returns:
    --------
    tasks: List[TinyImageNet]
        A list of TinyImageNet objects representing the tasks.
    """
    assert number_of_tasks in [5, 40], "There are only two available CL settings implemented: 5 tasks or 40 tasks"

    # Set randomly the order of classes
    rng = np.random.default_rng(seed)
    class_permutation = rng.permutation(200)
    handlers = []

    number_of_classes = int(200 / number_of_tasks)

    # 40 classification tasks with 5 classes in each or 5 tasks with 40 classes in each
    for i in range(0, number_of_classes * number_of_tasks, number_of_classes):
        current_labels = class_permutation[i:(i + number_of_classes)]
        print(f"Order of classes in the current task: {current_labels}")
        handlers.append(
            TinyImageNet(
                data_path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=use_one_hot,
                labels=current_labels
            )
        )

    return handlers


def prepare_toy_regression_tasks(
    seed: int = 1,
    no_of_validation_samples: int = 50, 
    number_of_tasks: int = 5
    ):
    """
    Prepare a list of tasks related to the 1D Regression Dataset (https://hypnettorch.readthedocs.io/en/latest/special.html#d-regression-dataset).

    Parameters:
    ----------
    seed: int
        Necessary for data generation
    no_of_validation_samples: int, optional
        The number of validation samples in each task. By default, it is 50.
    number_of_tasks: int, optional
        Defines the number of continual learning tasks. By default, it is 40.

    Returns:
    --------
    tasks: List[ToyRegression]
        A list of ToyRegression objects representing the tasks.
    """

    assert number_of_tasks == 5, "Number of tasks should be equal to 5."

    handlers = []
    domain = np.linspace(0, 1, number_of_tasks+1)

    # Generate functions
    # functions = [
    #     lambda x : x+3,
    #     lambda x : 2*x**2-1,
    #     lambda x : (x-3)**3,
    #     lambda x : np.sin(x),
    #     lambda x : np.tanh(x-8)
    # ]
    functions = [
        lambda x : 2*x,
        lambda x : 4*x**2-2,
        lambda x : 8*x**3-12*x,
        lambda x : 16*x**4 - 48*x**2 + 12,
        lambda x : 32*x**5 - 160*x**3 + 120*x
    ]

    # 'number_of_tasks' regression tasks
    for i in range(0, number_of_tasks):
        current_domain = domain[i:i+2]
        current_function = functions[i]
        handlers.append(
            ToyRegression(
                train_inter=current_domain,
                num_train=500,
                test_inter=current_domain,
                num_test=no_of_validation_samples,
                val_inter=current_domain,
                num_val=no_of_validation_samples,
                map_function=current_function,
                std=0.0,
                rseed=seed
            )
        )

    return handlers


def prepare_gaussian_regression_tasks(
    seed: int = 1,
    no_of_validation_samples: int = 50, 
    number_of_tasks: int = 5
    ):
    """
    Prepare a list of tasks related to the GaussianData
    (https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/data/special/gaussian_mixture_data.html#GaussianData).

    Parameters:
    ----------
    seed: int
        Necessary for data generation
    no_of_validation_samples: int, optional
        The number of validation samples in each task. By default, it is 50.
    number_of_tasks: int, optional
        Defines the number of continual learning tasks. By default, it is 40.

    Returns:
    --------
    tasks: List[GaussianData]
        A list of GaussianData objects representing the tasks.
    """

    assert number_of_tasks == 5, "Number of tasks should be equal to 5."
    
    # Generate Gaussians
    means = np.array([
        [5, 5],      # Gaussian 1
        [15, 15],    # Gaussian 2
        [25, 5],     # Gaussian 3
        [5, 25],     # Gaussian 4
        [25, 25]     # Gaussian 5
    ])

    covariances = [
        [[3, 1], [1, 2]],       # Covariance for Gaussian 1
        [[4, 1.5], [1.5, 3]],   # Covariance for Gaussian 2
        [[2, 0.5], [0.5, 1]],   # Covariance for Gaussian 3
        [[3, -1], [-1, 2]],     # Covariance for Gaussian 4
        [[4, -1.5], [-1.5, 3]]  # Covariance for Gaussian 5
    ]

    handlers = get_gmm_tasks(
        means=means,
        covs=covariances,
        num_train=500,
        num_test=no_of_validation_samples,
        num_val=no_of_validation_samples,
        rseed=seed
    )

    return handlers

if __name__ == "__main__":
    pass