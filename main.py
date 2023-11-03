import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from hypnettorch.mnets import MLP
from hypnettorch.mnets.resnet import ResNet
from hypnettorch.mnets.zenkenet import ZenkeNet
from hypnettorch.hnets import HMLP
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP
import hypnettorch.utils.hnet_regularizer as hreg
from torch import nn
from datetime import datetime
from itertools import product
from copy import deepcopy
from LearningTools.custom_loss_function import IBP_Loss
from Models.hmlp_ibp import HMLP_IBP

from datasets import (
    set_hyperparameters,
    prepare_split_cifar100_tasks,
    prepare_permuted_mnist_tasks,
    prepare_split_mnist_tasks
)

def set_seed(value):
    '''
    Set deterministic results according to the given value
    (including random, numpy and torch libraries)
    '''
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_row_to_file(filename, elements):
    '''
    Append a single row to the given file.

    Parameters
    ----------
    filename: folder and name of file
    elements: elements to saving in filename
    '''
    if not filename.endswith('.csv'):
        filename += '.csv'
    filename = filename.replace('.pt', '')
    with open(filename, "a+") as stream:
        np.savetxt(stream, np.array(elements)[np.newaxis], delimiter=';', fmt='%s')


def write_pickle_file(filename, object_to_save):
    torch.save(object_to_save, f'{filename}.pt')


def load_pickle_file(filename):
    return torch.load(filename)


def get_shapes_of_network(model):
    """
    Get shape of all layers of the loaded model.

    Argument:
    ---------
      *model*: an instance of hypnettorch model, e.g. MLP from mnets

    Returns:
    --------
      A list with lists of shapes of consecutive network layers
    """
    shapes_of_model = []
    for layer in model.weights:
        shapes_of_model.append(list(layer.shape))
    return shapes_of_model


def calculate_number_of_iterations(number_of_samples,
                                   batch_size,
                                   number_of_epochs):
    """
    Calculate the total number of iterations based on the number
    of samples, desired batch size and number of training epochs.

    Arguments:
    ----------
      *number_of_samples* (int) a number of individual samples
      *batch_size* (int) a number of samples entering the network
                   at one iteration
      *number_of_epochs* (int) a desired number of training epochs

    Returns:
    --------
      *no_of_iterations_per_epoch* (int) a number of training iterations
                                   per one epoch
      *total_no_of_iterations* (int) a total number of training iterations
    """
    no_of_iterations_per_epoch = int(
        np.ceil(number_of_samples / batch_size)
    )
    total_no_of_iterations = int(no_of_iterations_per_epoch * number_of_epochs)
    return no_of_iterations_per_epoch, total_no_of_iterations


def get_number_of_batch_normalization_layer(target_network):
    """
    Get a number of batch normalization layer in a given target network.
    Each normalization layer consists of two vectors.

    Arguments:
    ----------
      *target_network* (hypnettorch.mnets instance) a target network
      *target_network* (hypnettorch.mnets instance) a target network
    """
    if 'batchnorm_layers' in dir(target_network):
        if target_network.batchnorm_layers is None:
            num_of_batch_norm_layers = 0
        else:
            # Each layer contains a vector of means and a vector of
            # standard deviations
            num_of_batch_norm_layers = 2 * len(target_network.batchnorm_layers)
    else:
        num_of_batch_norm_layers = 0
    return num_of_batch_norm_layers


def calculate_accuracy(data,
                       target_network,
                       weights,
                       parameters,
                       evaluation_dataset):
    """
    Calculate accuracy for a given dataset using a selected network
    and a selected set of weights

    Arguments:
    ----------
      *data*: an instance of the dataset (e.g.
              hypnettorch.data.special.permuted_mnist.PermutedMNIST)
              in the case of the PermutedMNIST dataset
      *target_network*: an instance of the network that will be used
                        during calculations (not necessarily with weights)
      *weights*: weights for the *target_network* network
                 (an instance of torch.nn.modules.container.ParameterList)
      *parameters* a dictionary containing the following keys:
        -device- string: 'cuda' or 'cpu', defines in which device calculations
                 will be performed
        -use_batch_norm_memory- Boolean: defines whether stored weights
                                of the batch normalization layer should be used
                                If True then *number_of_task* has to be given
        -number_of_task- int/None: gives an information which task is currently
                         solved. The number must be given when
                         -use_batch_norm_memory- is True
      *evaluation_dataset*: (string) 'validation' or 'test'; defines whether
                            a validation or a test set will be evaluated
    Returns:
    --------
       torch.Tensor defining an accuracy for the selected setting
    """
    assert (parameters['use_batch_norm_memory'] and
            parameters['number_of_task'] is not None) or \
           not parameters['use_batch_norm_memory']
    assert evaluation_dataset in ['validation', 'test']
    target_network.eval()
    with torch.no_grad():
        # Currently results will be calculated on the validation or test set
        if evaluation_dataset == 'validation':
            input_data = data.get_val_inputs()
            output_data = data.get_val_outputs()
        elif evaluation_dataset == 'test':
            input_data = data.get_test_inputs()
            output_data = data.get_test_outputs()
        test_input = data.input_to_torch_tensor(
            input_data, parameters['device'], mode='inference'
        )
        test_output = data.output_to_torch_tensor(
            output_data, parameters['device'], mode='inference'
        )
        gt_classes = test_output.max(dim=1)[1]

        if parameters['use_batch_norm_memory']:
            logits = target_network.forward(
                test_input,
                weights=weights,
                condition=parameters['number_of_task']
            )
        else:
            logits = target_network.forward(
                test_input,
                weights=weights
            )
        predictions = logits.max(dim=1)[1]

        accuracy = (torch.sum(gt_classes == predictions, dtype=torch.float32) /
                    gt_classes.numel()) * 100.
    return accuracy

def evaluate_previous_tasks(hypernetwork,
                            target_network,
                            dataframe_results,
                            list_of_permutations,
                            parameters):
    """
    Evaluate the target network according to the weights generated
    by the hypernetwork for all previously trained tasks. For instance,
    if current_task_no is equal to 5, then tasks 0, 1, 2, 3, 4 and 5
    will be evaluated

    Arguments:
    ----------
      *hypernetwork* (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                     a hypernetwork that generates weights for the target
                     network
      *target_network* (hypnettorch.mnets module, e.g. mlp.MLP)
                       a target network that finally will perform
                       classification
      *dataframe_results* (Pandas Dataframe) stores results; contains
                          following columns: 'after_learning_of_task',
                          'tested_task' and 'accuracy'
      *list_of_permutations*: (hypnettorch.data module), e.g. in the case
                              of PermutedMNIST it will be
                              special.permuted_mnist.PermutedMNISTList
      *parameters* a dictionary containing the following keys:
        -device- string: 'cuda' or 'cpu', defines in which device calculations
                 will be performed
        -use_batch_norm_memory- Boolean: defines whether stored weights
                                of the batch normalization layer should be used
                                If True then *number_of_task* has to be given
        -number_of_task- int/None: gives an information which task is currently
                         solved

    Returns:
    --------
      *dataframe_results* (Pandas Dataframe) a dataframe updated with
                          the calculated results
    """
    # Calculate accuracy for each previously trained task
    # as well as for the last trained task
    hypernetwork.eval()
    target_network.eval()
    for task in range(parameters['number_of_task'] + 1):
        # Target entropy calculation should be included here: hypernetwork has to be inferred
        # for each task (together with the target network) and the task_id with the lowest entropy
        # has to be chosen
        # Arguments of the function: list of permutations, hypernetwork, sparsity, target network
        # output: task id
        currently_tested_task = list_of_permutations[task]
        # Generate weights of the target network
        target_weights = hypernetwork.forward(cond_id=task)
        
        accuracy = calculate_accuracy(
            currently_tested_task,
            target_network,
            target_weights,
            parameters=parameters,
            evaluation_dataset='test'
        )
        result = {
            'after_learning_of_task': parameters['number_of_task'],
            'tested_task': task,
            'accuracy': accuracy.cpu().item()
        }
        print(f'Accuracy for task {task}: {accuracy}%.')
        dataframe_results = dataframe_results.append(
            result, ignore_index=True)
    return dataframe_results


def save_parameters(saving_folder,
                    parameters,
                    name=None):
    """
    Save hyperparameters to the selected file.

    Arguments:
    ----------
      *saving_folder* (string) defines a path to the folder for saving
      *parameters* (dictionary) contains all hyperparameters to saving
      *name* (optional string) name of the file for saving
    """
    if name is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f'parameters_{current_time}.csv'
    with open(f'{saving_folder}/{name}', 'w') as file:
        for key in parameters.keys():
            file.write(f'{key};{parameters[key]}\n')


def plot_heatmap(load_path):
    """
    Plot heatmap presenting results for different learning tasks

    Argument:
    ---------
      *load_path* (string) contains path to the .csv file with
                  results in a dataframe shape, i.e. with columns:
                  'after_learning_of_task', 'tested_task' (both
                  integers) and 'accuracy' (float)
    """
    dataframe = pd.read_csv(load_path, delimiter=';', index_col=0)
    dataframe = dataframe.astype(
        {'after_learning_of_task': 'int32',
         'tested_task': 'int32'}
        )
    table = dataframe.pivot(
        'after_learning_of_task', 'tested_task', 'accuracy')
    sns.heatmap(table, annot=True, fmt='.1f')
    plt.tight_layout()
    plt.savefig(load_path.replace('.csv', '.pdf'), dpi=300)
    plt.close()

def plot_intervals_around_embeddings(tasks_embeddings,
                                     trained_radii,
                                     save_folder,
                                     no_of_task,
                                     n_embs_to_plot=20):
    """
    Plot intervals with trained radii around tasks' embeddings

    Argument:
    ---------
        *task_embeddings*: (torch.Tensor) contains tasks' embeddings
        *trained_radii*: (torch.Tensor) contains trained radii around
                        tasks' embeddings
        *save_folder*: (string) contains folder where the plot will be saved,
        *no_of taks*: (int) number of currently learned task
        *n_embs_to_plot*: (int) number of embeddings to be plotted
    """
    
    # Check if these two tensors have the same length
    assert len(tasks_embeddings) == len(trained_radii)

    # Check if folder exists, if it doesn't then create the folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Send tensors to the CPU device and convert into numpy objects
    tasks_embeddings = tasks_embeddings.cpu().detach().numpy()[0][:n_embs_to_plot]
    trained_radii    = trained_radii.cpu().detach().numpy()[0][:n_embs_to_plot]

    # Generate an x axis
    x = [_ for _ in range(len(tasks_embeddings))]

    # Create a scatter plot
    plt.scatter(x, tasks_embeddings, color='blue', marker='o')

    # Draw horizontal lines around the dots
    for i in range(len(x)):
        plt.vlines(tasks_embeddings[i], ymin=x[i] - trained_radii[i],
                    ymax=x[i] + trained_radii[i], colors='red', linewidth=2)

    # Create a save path
    save_path = f'{save_folder}/intervals_around_tasks_embeddings_{no_of_task}.png'

    # Add labels and a legend
    plt.xlabel("Embedding's coordinate")
    plt.ylabel('Radii')
    plt.title('Intervals around embeddings')
    plt.xticks(x)
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

def train_single_task(hypernetwork,
                      target_network,
                      criterion,
                      parameters,
                      dataset_list_of_tasks,
                      current_no_of_task):
    """
    Train two neural networks: a hypernetwork will generate the weights of the target neural network.
    This module operates on a single training task with a specific number.

    Arguments:
    ----------
      *hypernetwork*: (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                      a hypernetwork that generates weights for the target
                      network
      *target_network*: (hypnettorch.mnets module, e.g. mlp.MLP)
                        a target network that finally will perform
                        classification
      *criterion*: (torch.nn module) implements a loss function,
                   e.g. CrossEntropyLoss
      *parameters*: (dictionary) contains necessary hyperparameters
                    describing an experiment
      *dataset_list_of_tasks*: a module containing list of tasks for the CL
                               scenario, e.g. permuted_mnist.PermutedMNISTList
      *current_no_of_task*: (int) specifies the number of currently solving task

    Returns:
    --------
      *hypernetwork*: a modified module of hypernetwork
      *target_network*: a modified module of the target network
    """
    # Optimizer cannot be located outside of this function because after
    # deep copy of the network it needs to be reinitialized
    if parameters["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters['learning_rate']
        )
    elif parameters["optimizer"] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters['learning_rate']
        )
    else:
        raise ValueError('Wrong type of the selected optimizer!')
    if parameters['best_model_selection_method'] == 'val_loss':
        # Store temporary best models to keep those with the highest
        # validation accuracy.
        best_hypernetwork = deepcopy(hypernetwork)
        best_target_network = deepcopy(target_network)
        best_val_accuracy = 0.
    elif parameters['best_model_selection_method'] != 'last_model':
        raise ValueError('Wrong value of best_model_selection_method parameter!')
    # Compute targets for the regularization part of loss before starting
    # the training of a current task
    hypernetwork.train()
    target_network.train()
    print(f'task: {current_no_of_task}')
    if current_no_of_task > 0:
        regularization_targets = hreg.get_current_targets(
            current_no_of_task, hypernetwork)
        previous_hnet_theta = None
        previous_hnet_embeddings = None
        previous_target_weights = deepcopy(target_network.weights)
    else:
        previous_target_weights = None

    if (parameters['target_network'] == 'ResNet') and \
       parameters['use_batch_norm']:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
    # If training through a given number of epochs is desired
    # the number of iterations has to be calculated
    if parameters['number_of_epochs'] is not None:
        no_of_iterations_per_epoch, parameters['number_of_iterations'] = \
            calculate_number_of_iterations(
                current_dataset_instance.num_train_samples,
                parameters['batch_size'],
                parameters['number_of_epochs']
            )
        # Scheduler can be set only when the number of epochs is given
        if parameters['lr_scheduler']:
            current_epoch = 0
            plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'max', factor=np.sqrt(0.1), patience=5,
                min_lr=0.5e-6, cooldown=0, verbose=True
            )
    # The kappa and epsilon hyperparameters scheduling
    iterations_to_adjust = hyperparameters["number_of_iterations"] // 2

    for iteration in range(parameters['number_of_iterations']):
        current_batch = current_dataset_instance.next_train_batch(
            parameters['batch_size']
        )
        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], parameters['device'], mode='train'
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], parameters['device'], mode='train'
        )
        gt_output = tensor_output.max(dim=1)[1]
        optimizer.zero_grad()

        # Adjust kappa and epsilon
        if iteration < iterations_to_adjust:
            kappa = max(1 - 0.00005*iteration, hyperparameters["kappa"])
            eps   = iteration / (iterations_to_adjust-1) * hyperparameters["perturbated_epsilon"]
        else:
            kappa = hyperparameters["kappa"]
            eps   = hyperparameters["perturbated_epsilon"]

        # Get weights, lower weights, upper weights and predicted radii
        # returned by the hypernetwork
        target_weights, lower_weights, upper_weights, radii = hypernetwork.forward(cond_id=current_no_of_task, 
                                                                            return_extended_output=True,
                                                                            perturbated_eps=eps)
        
        loss_norm_target_regularizer = 0.
        if current_no_of_task > 0:
            # Add another regularizer for weights, e.g. according
            # to L1 or L2 norm. The goal is to prevent significant
            # changes in weights between consecutive tasks.
            # ATTENTION! This norm is not calculated for batch
            # normalization layers
            no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
                target_network)
            for no_of_layer in range(len(list(target_network.children()))):
                loss_norm_target_regularizer += torch.norm(
                    target_network.weights[
                        no_of_layer + no_of_batch_norm_layers] -
                    previous_target_weights[
                        no_of_layer + no_of_batch_norm_layers],
                    p=parameters['norm']
                )
    

        # Even if batch normalization layers are applied, statistics
        # for the last saved tasks will be applied so there is no need to
        # give 'current_no_of_task' as a value for the 'condition' argument.
        prediction = target_network.forward(tensor_input,
                                            weights=target_weights)
        prediction_zu = target_network.forward(tensor_input,
                                                weights=upper_weights)
        prediction_zl = target_network.forward(tensor_input,
                                                weights=lower_weights)
        

        loss_current_task = criterion(
            y_pred=prediction,
            y=gt_output,
            z_l=prediction_zl,
            z_u=prediction_zu,
            radii=radii,
            eps=eps,
            kappa=kappa
        )

        loss_regularization = 0.
        if current_no_of_task > 0:
            loss_regularization = hreg.calc_fix_target_reg(
                hypernetwork, current_no_of_task,
                targets=regularization_targets,
                mnet=target_network, prev_theta=previous_hnet_theta,
                prev_task_embs=previous_hnet_embeddings,
                inds_of_out_heads=None,
                batch_size=-1
            )
        append_row_to_file(
            f'{parameters["saving_folder"]}regularization_loss.csv',
            f'{current_no_of_task};{iteration};'
            f'{loss_regularization};{loss_norm_target_regularizer}'
        )
        loss = loss_current_task + \
            parameters['beta'] * loss_regularization / max(1, current_no_of_task) + \
            parameters['lambda'] * loss_norm_target_regularizer
        #  print(f'loss: {loss}')
        loss.backward()
        optimizer.step()
        if parameters['number_of_epochs'] is None:
            condition = (iteration % 100 == 0) or \
                        (iteration == (parameters['number_of_iterations'] - 1))
        else:
            condition = (iteration % 100 == 0) or \
                        (iteration == (parameters['number_of_iterations'] - 1)) or \
                        (((iteration + 1) % no_of_iterations_per_epoch) == 0)

        if condition:
            if parameters['number_of_epochs'] is not None:
                current_epoch = (iteration + 1) // no_of_iterations_per_epoch
                print(f'Current epoch: {current_epoch}')
            accuracy = calculate_accuracy(
                current_dataset_instance,
                target_network,
                target_weights,
                parameters={
                    'device': parameters['device'],
                    'use_batch_norm_memory': use_batch_norm_memory,
                    'number_of_task': current_no_of_task
                },
                evaluation_dataset='validation')
            
            # Get the worst case error
            worst_case_error = criterion.worst_case_error

            print(f'Task {current_no_of_task}, iteration: {iteration + 1},'
                  f' loss: {loss.item()}, validation accuracy: {accuracy},'
                  f' worst case error: {worst_case_error},'
                  f' predicted radii mean: {radii.mean().item()}')
            # If the accuracy on the validation dataset is higher
            # than previously
            if parameters['best_model_selection_method'] == 'val_loss':
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    best_hypernetwork = deepcopy(hypernetwork)
                    best_target_network = deepcopy(target_network)
            
            if parameters['number_of_epochs'] is not None and \
               parameters['lr_scheduler'] and \
               (((iteration + 1) % no_of_iterations_per_epoch) == 0):
                print('Finishing the current epoch')
                # scheduler.step()
                plateau_scheduler.step(accuracy)

    if parameters['best_model_selection_method'] == 'val_loss':
        return best_hypernetwork, best_target_network
    else:
        return hypernetwork, target_network


def build_multiple_task_experiment(dataset_list_of_tasks,
                                   parameters,
                                   use_chunks=False):
    """
    Create a continual learning experiment with multiple tasks
    for a given dataset.

    Arguments:
    ----------
      *dataset_list_of_tasks*: a module containing list of tasks for the CL
                               scenario, e.g. permuted_mnist.PermutedMNISTList
      *parameters*: (dictionary) contains necessary hyperparameters
                    describing an experiment
      *use_chunks*: (Boolean value) optional argument, defines whether
                    a hypernetwork should generate weights in chunks or not

    Returns:
    --------
      *hypernetwork*: (hypnettorch.hnets module, e.g. mlp_hnet.MLP)
                      a hypernetwork that generates weights for the target
                      network
      *target_network*: (hypnettorch.mnets module, e.g. mlp.MLP)
                        a target network that finally will perform
                        classification
      *dataframe*: (Pandas Dataframe) contains results from consecutive
                   evaluations for all previous tasks
    """
    output_shape = list(
        dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]
    # Create a target network which will be multilayer perceptron
    # or ResNet/ZenkeNet with internal weights
    if parameters['target_network'] == 'MLP':
        target_network = MLP(n_in=parameters['input_shape'],
                             n_out=output_shape,
                             hidden_layers=parameters['target_hidden_layers'],
                             use_bias=parameters['use_bias'],
                             no_weights=False).to(parameters['device'])
    elif parameters['target_network'] == 'ResNet':
        target_network = ResNet(in_shape=(parameters['input_shape'],
                                          parameters['input_shape'],
                                          3),
                                use_bias=parameters['use_bias'],
                                num_classes=output_shape,
                                n=parameters['resnet_number_of_layer_groups'],
                                k=parameters['resnet_widening_factor'],
                                no_weights=False,
                                use_batch_norm=parameters['use_batch_norm'],
                                bn_track_stats=False).to(parameters['device'])
    elif parameters['target_network'] == 'ZenkeNet':
        target_network = ZenkeNet(in_shape=(parameters['input_shape'],
                                            parameters['input_shape'],
                                            3),  
                                  num_classes=output_shape,
                                  arch='cifar',
                                  no_weights=False).to(parameters['device'])
    # Create a hypernetwork based on the shape of the target network
    no_of_batch_norm_layers = get_number_of_batch_normalization_layer(
        target_network
    )
    if not use_chunks:
        hypernetwork = HMLP_IBP(
            perturbated_eps=parameters['perturbated_epsilon'],
            target_shapes=target_network.param_shapes[no_of_batch_norm_layers:],
            uncond_in_size=0,
            cond_in_size=parameters['embedding_size'],
            activation_fn=parameters['activation_function'],
            layers=parameters['hypernetwork_hidden_layers'],
            num_cond_embs=parameters['number_of_tasks']).to(
                parameters['device'])
    else:
        raise Exception("Not implemented yet!")
        # hypernetwork = ChunkedHMLP(
        #     target_shapes=target_network.param_shapes[no_of_batch_norm_layers:],
        #     chunk_size=parameters['chunk_size'],
        #     chunk_emb_size=parameters['chunk_emb_size'],
        #     cond_in_size=parameters['embedding_size'],
        #     activation_fn=parameters['activation_function'],
        #     layers=parameters['hypernetwork_hidden_layers'],
        #     num_cond_embs=parameters['number_of_tasks']).to(
        #         parameters['device']
        # )

    criterion = IBP_Loss(calculation_area_mode=parameters['calculation_area_mode'])
    dataframe = pd.DataFrame(columns=[
        'after_learning_of_task', 'tested_task', 'accuracy'])

    if (parameters['target_network'] == 'ResNet') and \
       parameters['use_batch_norm']:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    hypernetwork.train()
    target_network.train()
    for no_of_task in range(parameters['number_of_tasks']):
        hypernetwork, target_network = train_single_task(
            hypernetwork,
            target_network,
            criterion,
            parameters,
            dataset_list_of_tasks,
            no_of_task
        )

        if no_of_task == (parameters['number_of_tasks'] - 1):
        # Save current state of networks
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f'hypernetwork_after_{no_of_task}_task',
                hypernetwork.weights
            )
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f'target_network_after_{no_of_task}_task',
                target_network.weights
            )

        # Create intervals over tasks' embeddings plot
        interval_plot_save_path = f'{parameters["saving_folder"]}/plots/'
        
        plot_intervals_around_embeddings(tasks_embeddings=hypernetwork.tasks_embeddings,
                                     trained_radii=hypernetwork.trained_radii,
                                     save_folder=interval_plot_save_path,
                                     no_of_task=no_of_task)
        
        dataframe = evaluate_previous_tasks(
            hypernetwork,
            target_network,
            dataframe,
            dataset_list_of_tasks,
            parameters={
                'device': parameters['device'],
                'use_batch_norm_memory': use_batch_norm_memory,
                'number_of_task': no_of_task
            }
        )
        dataframe = dataframe.astype({
            'after_learning_of_task': 'int',
            'tested_task': 'int'
        })
        dataframe.to_csv(f'{parameters["saving_folder"]}/'
                         f'results.csv',
                         sep=';')
    return hypernetwork, target_network, dataframe


def main_running_experiments(path_to_datasets,
                             parameters,
                             dataset):
    """
    Perform a series of experiments based on the hyperparameters.

    Arguments:
    ----------
      *path_to_datasets*: (str) path to files with datasets
      *parameters*: (dict) contains multiple experiment hyperparameters
      *dataset*: (str) dataset for calculation: PermutedMNIST,
                 CIFAR100 or SplitMNIST

    Returns learned hypernetwork, target network and a dataframe
    with single results.
    """
    if dataset == 'PermutedMNIST':
        dataset_tasks_list = prepare_permuted_mnist_tasks(
            path_to_datasets,
            parameters["input_shape"],
            parameters["number_of_tasks"],
            parameters["padding"],
            parameters["no_of_validation_samples"]
        )
    elif dataset == 'CIFAR100':
        dataset_tasks_list = prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=parameters['no_of_validation_samples'],
            use_augmentation=parameters['augmentation']
        )
    elif dataset == 'SplitMNIST':
        dataset_tasks_list = prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=parameters['no_of_validation_samples'],
            use_augmentation=parameters['augmentation'],
            number_of_tasks=parameters['number_of_tasks'],
        )
    else:
        raise ValueError('Wrong name of the dataset!')

    hypernetwork, target_network, dataframe = build_multiple_task_experiment(
        dataset_tasks_list,
        parameters,
        use_chunks=parameters['use_chunks']
    )
    # Calculate statistics of grid search results
    no_of_last_task = parameters["number_of_tasks"] - 1
    accuracies = dataframe.loc[
        dataframe['after_learning_of_task'] == no_of_last_task
    ]['accuracy'].values
    row_with_results = (
        f'{dataset_tasks_list[0].get_identifier()};'
        f'{parameters["augmentation"]};'
        f'{parameters["embedding_size"]};'
        f'{parameters["seed"]};'
        f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
        f'{parameters["use_chunks"]};{parameters["chunk_emb_size"]};'
        f'{parameters["target_network"]};'
        f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
        f'{parameters["resnet_number_of_layer_groups"]};'
        f'{parameters["resnet_widening_factor"]};'
        f'{parameters["best_model_selection_method"]};'
        f'{parameters["optimizer"]};'
        f'{parameters["activation_function"]};'
        f'{parameters["learning_rate"]};{parameters["batch_size"]};'
        f'{parameters["beta"]};'
        f'{parameters["norm"]};{parameters["lambda"]};'
        f'{np.mean(accuracies)};{np.std(accuracies)}'
    )
    append_row_to_file(
        f'{parameters["grid_search_folder"]}'
        f'{parameters["summary_results_filename"]}.csv',
        row_with_results
    )

    load_path = (f'{parameters["saving_folder"]}/'
                 f'results.csv')

    plot_heatmap(load_path)
    
    return hypernetwork, target_network, dataframe


if __name__ == "__main__":
    path_to_datasets = './Data'
    dataset = 'PermutedMNIST'  # 'PermutedMNIST', 'CIFAR100', 'SplitMNIST'
    part = 2
    create_grid_search = False
    if create_grid_search:
        summary_results_filename = 'grid_search_results'
    else:
        summary_results_filename = 'summary_results'
    hyperparameters = set_hyperparameters(
        dataset,
        grid_search=create_grid_search,
        part=part
    )
    header = (
        'dataset_name;augmentation;embedding_size;seed;hypernetwork_hidden_layers;'
        'use_chunks;chunk_emb_size;target_network;target_hidden_layers;'
        'layer_groups;widening;final_model;optimizer;'
        'hypernet_activation_function;learning_rate;batch_size;beta;norm;lambda;mean_accuracy;std_accuracy'
    )

    append_row_to_file(
        f'{hyperparameters["saving_folder"]}{summary_results_filename}.csv',
        header
    )

    for no, elements in enumerate(
        product(hyperparameters["embedding_sizes"],
                hyperparameters["learning_rates"],
                hyperparameters["betas"],
                hyperparameters["hypernetworks_hidden_layers"],
                hyperparameters["lambdas"],
                hyperparameters["batch_sizes"],
                hyperparameters["seed"])
    ):
        embedding_size = elements[0]
        learning_rate = elements[1]
        beta = elements[2]
        hypernetwork_hidden_layers = elements[3]
        lambda_par = elements[4]
        batch_size = elements[5]
        # Of course, seed is not optimized but it is easier to prepare experiments
        # for multiple seeds in such a way
        seed = elements[6]

        parameters = {
            'input_shape': hyperparameters["shape"],
            'augmentation': hyperparameters["augmentation"],
            'number_of_tasks': hyperparameters["number_of_tasks"],
            'seed': seed,
            'hypernetwork_hidden_layers': hypernetwork_hidden_layers,
            'activation_function': hyperparameters["activation_function"],
            'use_chunks': hyperparameters["use_chunks"],
            'chunk_size': hyperparameters["chunk_size"],
            'chunk_emb_size': hyperparameters["chunk_emb_size"],
            'target_network': hyperparameters["target_network"],
            'target_hidden_layers': hyperparameters["target_hidden_layers"],
            'resnet_number_of_layer_groups': hyperparameters["resnet_number_of_layer_groups"],
            'resnet_widening_factor': hyperparameters["resnet_widening_factor"],
            'learning_rate': learning_rate,
            'best_model_selection_method': hyperparameters['best_model_selection_method'],
            'lr_scheduler': hyperparameters["lr_scheduler"],
            'batch_size': batch_size,
            'number_of_epochs': hyperparameters["number_of_epochs"],
            'number_of_iterations': hyperparameters["number_of_iterations"],
            'no_of_validation_samples': hyperparameters["no_of_validation_samples"],
            'embedding_size': embedding_size,
            'norm': hyperparameters["norm"],
            'lambda': lambda_par,
            'optimizer': hyperparameters["optimizer"],
            'beta': beta,
            'padding': hyperparameters["padding"],
            'use_bias': hyperparameters["use_bias"],
            'use_batch_norm': hyperparameters["use_batch_norm"],
            'device': hyperparameters["device"],
            'saving_folder': f'{hyperparameters["saving_folder"]}{no}/',
            'grid_search_folder': hyperparameters["saving_folder"],
            'summary_results_filename': summary_results_filename,
            'calculation_area_mode': hyperparameters["calculation_area_mode"],
            'perturbated_epsilon': hyperparameters["perturbated_epsilon"],
            'kappa': hyperparameters["kappa"]
        }

        os.makedirs(parameters['saving_folder'], exist_ok=True)
        # start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_parameters(parameters["saving_folder"],
                        parameters,
                        name=f'parameters.csv')

        # Important! Seed is set before the preparation of the dataset!
        if seed is not None:
            set_seed(seed)

        hypernetwork, target_network, dataframe = \
            main_running_experiments(path_to_datasets,
                                     parameters,
                                     dataset)
