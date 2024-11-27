"""
Split CUB200 Dataset
^^^^^^^^^^^^^^^^^^^

The module contains a wrapper for data handlers for the SplitCUB200 task.
"""

from torchvision import transforms
import torchvision.datasets as datasets
import torch

import os
import time
import urllib.request
import tarfile
import pandas
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hypnettorch.data.large_img_dataset import LargeImgDataset
from hypnettorch.data.ilsvrc2012_data import ILSVRC2012Data
from hypnettorch.data.dataset import Dataset

def _transform_split_outputs(data, outputs):
    """Actual implementation of method ``transform_outputs`` for split dataset
    handlers.

    Paramaters:
    -----------
        data: Iterable[np.ndarray]
            Data handler.
        outputs: np.ndarray
            
    
    Returns:
    ---------
        (numpy.ndarray)
    """
    if not data._full_out_dim:
        # TODO implement reverse direction as well.
        raise NotImplementedError('This method is currently only ' +
            'implemented if constructor argument "full_out_dim" was set.')

    labels = data._labels
    if data.is_one_hot:
        assert(outputs.shape[1] == data._data['num_classes'])
        mask = np.zeros(data._data['num_classes'], dtype=np.bool)
        mask[labels] = True

        return outputs[:, mask]
    else:
        assert (outputs.shape[1] == 1)
        ret = outputs.copy()
        for i, l in enumerate(labels):
            ret[ret == l] = i
        return ret


def get_split_cub200_handlers(data_path, use_one_hot=True, validation_size=0,
                             use_torch_augmentation=False,
                             num_classes_per_task=10, num_tasks=None,
                             trgt_padding=None):
    """This function instantiates 20 objects of the class :class:`SplitCUB200`
    which will contain a disjoint set of labels.

    The SplitCUB200 task consists of 20 tasks corresponding to the images with
    labels {0,...,9}, ..., {190, 199}.

    Paramaters:
    ------------
        data_path: str
            Where should the CUB200 dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: bool
            Whether the class labels should be represented in a one-hot
            encoding.
        validation_size: int
            The size of the validation set of each individual
            data handler.
        use_torch_augmentation: bool
            Flag to indicate whether data augmentation should be used or not.
        num_classes_per_task: int
            Number of classes to put into one data handler. 
            If ``2``, then every data handler will include 2 digits.
        num_tasks: int, optional
            The number of data handlers that should be returned by this function.
        trgt_padding: int, optional
            See docstring of class :class:`SplitCUB200`.

    Returns:
    ---------  
        A list of data handlers, each corresponding to a
        :class:`SplitCUB200` object.
    """
    assert num_tasks is None or num_tasks > 0
    if num_tasks is None:
        num_tasks = 200 // num_classes_per_task

    if not (num_tasks >= 1 and (num_tasks * num_classes_per_task) <= 200):
        raise ValueError('Cannot create SplitCUB200 datasets for %d tasks ' \
                         % (num_tasks) + 'with %d classes per task.' \
                         % (num_classes_per_task))

    print('Creating %d data handlers for SplitCUB200 tasks ...' % num_tasks)

    handlers = []
    steps = num_classes_per_task
    for i in range(0, 200, steps):
        handlers.append(SplitCUB200Data(data_path, use_one_hot=use_one_hot,
            use_torch_augmentation=use_torch_augmentation,
            validation_size=validation_size, labels=range(i, i+steps),
            trgt_padding=trgt_padding))

        if len(handlers) == num_tasks:
            break

    print('Creating data handlers for SplitCUB200 tasks ... Done')

    return handlers


class CUB2002011(LargeImgDataset):
    """An instance of the class shall represent the CUB-200-2011 dataset.

    The input data of the dataset will be strings to image files. The output
    data corresponds to object labels (bird categories).

    Note:
        The dataset will be downloaded if not available.

    Note:
        The original category labels range from 1-200. We modify them to
        range from 0 - 199.

    Args:
        data_path (str): Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.

            .. note::
                This option does not influence the internal PyTorch
                Dataset classes (e.g., cmp.
                :attr:`data.large_img_dataset.LargeImgDataset.torch_train`),
                that can be used in conjunction with PyTorch data loaders.
        num_val_per_class (int): The number of validation samples per class.
            For instance: If value 10 is given, a validation set of size
            5 * 200 = 1,000 is constructed (these samples will be removed
            from the training set).

            .. note::
                Validation samples use the same data augmentation pipeline
                as test samples.
    """
    _DOWNLOAD_PATH = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    _IMG_ANNO_FILE = 'CUB_200_2011.tgz'
    _SEGMENTATION_FILE = 'segmentations.tgz' # UNUSED
    # In which subfolder of the datapath should the data be stored.
    _SUBFOLDER = 'cub_200_2011'
    # After extracting the downloaded archive, the data will be in
    # this subfolder.
    _REL_BASE = 'CUB_200_2011'
    _IMG_DIR = 'images' # Realitve to _REL_BASE
    _CLASSES_FILE = 'classes.txt' # Realitve to _REL_BASE
    _IMG_CLASS_LBLS_FILE = 'image_class_labels.txt' # Realitve to _REL_BASE
    _IMG_FILE = 'images.txt' # Realitve to _REL_BASE
    _TRAIN_TEST_SPLIT_FILE = 'train_test_split.txt' # Realitve to _REL_BASE


    def __init__(self, data_path, use_one_hot=False, num_val_per_class=0):
        # We keep the full path to each image in memory, so we don't need to
        # tell the super class the root path to each image (i.e., samples
        # contain absolute not relative paths).


        super().__init__('')

        self.test_transform = transforms.Compose(
            [
                transforms.Resize((32,32))
            ]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((32,32))
            ]
        )

        start = time.time()

        print('Reading CUB-200-2011 dataset ...')

        # Actual data path
        data_path = os.path.join(data_path, CUB2002011._SUBFOLDER)

        if not os.path.exists(data_path):
            print('Creating directory "%s" ...' % (data_path))
            os.makedirs(data_path)
            
        full_data_path = os.path.join(data_path, CUB2002011._REL_BASE)
        image_dir = os.path.join(full_data_path, CUB2002011._IMG_DIR)
        classes_fn = os.path.join(full_data_path, CUB2002011._CLASSES_FILE)
        img_class_fn = os.path.join(full_data_path,
                                    CUB2002011._IMG_CLASS_LBLS_FILE)
        image_fn = os.path.join(full_data_path, CUB2002011._IMG_FILE)
        train_test_split_fn = os.path.join(full_data_path,
                                           CUB2002011._TRAIN_TEST_SPLIT_FILE)

        ########################
        ### Download dataset ###
        ########################
        if not os.path.exists(image_dir) or \
                not os.path.exists(classes_fn) or \
                not os.path.exists(img_class_fn) or \
                not os.path.exists(image_fn) or \
                not os.path.exists(train_test_split_fn):
            print('Downloading dataset ...')
            archive_fn = os.path.join(data_path, CUB2002011._IMG_ANNO_FILE)
            urllib.request.urlretrieve(CUB2002011._DOWNLOAD_PATH + \
                                       CUB2002011._IMG_ANNO_FILE, \
                                       archive_fn)
            # Extract downloaded dataset.
            tar = tarfile.open(archive_fn, "r:gz")
            tar.extractall(path=data_path)
            tar.close()

            os.remove(archive_fn)

        ####################
        ### Read dataset ###
        ####################
        # We use the same transforms as 
        train_transform, test_transform = \
            ILSVRC2012Data.torch_input_transforms()
        # Consider all images as training images. We split the dataset later.
        ds_train = datasets.ImageFolder(image_dir, train_transform)

        # Ability to translate image IDs into image paths and back.
        image_ids_csv = pandas.read_csv(image_fn, sep=' ',
                                        names=['img_id', 'img_path'])
        id2img = dict(zip(list(image_ids_csv['img_id']),
                          list(image_ids_csv['img_path'])))
        # Since the ImageFolder class uses absolute paths, we have to change
        # the just read relative paths.
        for iid in id2img.keys():
            id2img[iid] = os.path.join(image_dir, id2img[iid])
        img2id = {v: k for k, v in id2img.items()}

        # Image ID to label.
        img_lbl_csv = pandas.read_csv(img_class_fn, sep=' ',
                                      names=['img_id', 'label'])
        id2lbl = dict(zip(list(img_lbl_csv['img_id']),
                          list(img_lbl_csv['label'])))
        # Note, categories go from 1-200. We change them to go from 0 - 199.
        for iid in id2lbl.keys():
            id2lbl[iid] = id2lbl[iid] - 1

        # Image ID to label name.
        img_lbl_name_csv = pandas.read_csv(classes_fn, sep=' ',
                                           names=['label', 'label_name'])
        lbl2lbl_name_tmp = dict(zip(list(img_lbl_name_csv['label']),
                                    list(img_lbl_name_csv['label_name'])))
        # Here, we also have to modify the labels to be within 0-199.
        lbl2lbl_name = {k-1: v for k, v in lbl2lbl_name_tmp.items()}

        # Train-test-split.
        train_test_csv = pandas.read_csv(train_test_split_fn, sep=' ',
                                         names=['img_id', 'is_train'])
        id2train = dict(zip(list(train_test_csv['img_id']),
                            list(train_test_csv['is_train'])))

        self._label_to_name = lbl2lbl_name

        ####################
        ### Sanity check ###
        ####################
        for i, (img_path, lbl) in enumerate(ds_train.samples):
            iid = img2id[img_path]
            assert(id2img[iid] == img_path)
            assert(lbl == id2lbl[iid])

        ################################
        ### Train / val / test split ###
        ################################
        orig_samples = ds_train.samples
        ds_train.samples = []
        ds_train.imgs = ds_train.samples
        ds_train.targets = []

        ds_test = deepcopy(ds_train)
        ds_test.transform = test_transform
        assert(ds_test.target_transform is None)
        if num_val_per_class > 0:
            ds_val = deepcopy(ds_train)
            # NOTE we use test input transforms for the validation set.
            ds_val.transform = test_transform
        else:
            ds_val = None

        num_classes = len(lbl2lbl_name_tmp.keys())
        assert(num_classes == 200)
        val_counts = np.zeros(num_classes, dtype=np.int)

        for img_path, img_lbl in orig_samples:
            iid = img2id[img_path]
            if id2train[iid] == 1: # In train split.
                if val_counts[img_lbl] >= num_val_per_class: # train sample
                    ds_train.samples.append((img_path, img_lbl))
                else: # validation sample
                    val_counts[img_lbl] += 1
                    ds_val.samples.append((img_path, img_lbl))
            else: # In test split.
                ds_test.samples.append((img_path, img_lbl))

        for ds_obj in [ds_train, ds_test] + \
                ([ds_val] if num_val_per_class > 0 else []):
            ds_obj.targets = [s[1] for s in ds_obj.samples]
            assert(len(ds_obj.samples) == len(ds_obj.imgs) and \
                   len(ds_obj.samples) == len(ds_obj.targets))

        self._torch_ds_train = ds_train
        self._torch_ds_test = ds_test
        self._torch_ds_val = ds_val

        #####################################
        ### Build internal data structure ###
        #####################################
        num_train = len(self._torch_ds_train)
        num_test = len(self._torch_ds_test)
        num_val = 0 if self._torch_ds_val is None else \
            len(self._torch_ds_val)
        num_samples = num_train + num_test + num_val

        max_path_len = len(max(orig_samples, key=lambda t : len(t[0]))[0])

        self._data['classification'] = True
        self._data['sequence'] = False
        self._data['num_classes'] = 200
        self._data['is_one_hot'] = use_one_hot

        self._data['in_shape'] = [32, 32, 3]
        self._data['out_shape'] = [200 if use_one_hot else 1]

        self._data['in_data'] = np.chararray([num_samples, 1],
            itemsize=max_path_len, unicode=True)
        for i, (img_path, _) in enumerate(ds_train.samples +
                ([] if num_val == 0 else ds_val.samples) +
                ds_test.samples):
            self._data['in_data'][i, :] = img_path

        labels = np.array(ds_train.targets +
                          ([] if num_val == 0 else ds_val.targets) +
                          ds_test.targets).reshape(-1, 1)
        if use_one_hot:
            labels = self._to_one_hot(labels)
        self._data['out_data'] = labels

        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train + num_val, num_samples)
        if num_val == 0:
            self._data['val_inds'] = None
        else:
            self._data['val_inds'] = np.arange(num_train, num_train + num_val)

        print('Dataset consists of %d training, %d validation and %d test '
              % (num_train, num_val, num_test) + 'samples.')

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'CUB-200-2011'

    def tf_input_map(self, mode='inference'):
        """Not impemented."""
        # Confirm, whether you wanna process data as in the baseclass or
        # implement a new image loader.
        raise NotImplementedError('Not implemented yet!')

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        """Implementation of abstract method
        :meth:`data.dataset.Dataset._plot_sample`.
        """
        ax = plt.Subplot(fig, inner_grid[0])

        if outputs is None:
            ax.set_title("CUB-200-2011 Sample")
        else:
            assert(np.size(outputs) == 1)
            label = np.asscalar(outputs)
            label_name = self._label_to_name[label]

            if predictions is None:
                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label))
            else:
                if np.size(predictions) == self.num_classes:
                    pred_label = np.argmax(predictions)
                else:
                    pred_label = np.asscalar(predictions)
                pred_label_name = self._label_to_name[pred_label]

                ax.set_title('Label of shown sample:\n%s (%d)' % \
                             (label_name, label) + '\nPrediction: %s (%d)' % \
                             (pred_label_name, pred_label))

        if inputs.size == 1:
            img = self.read_images(inputs)
        else:
            img = inputs

        ax.set_axis_off()
        ax.imshow(np.squeeze(np.reshape(img, self.in_shape)))
        fig.add_subplot(ax)

    def input_to_torch_tensor(
        self,
        x,
        device,
        mode="inference",
        force_no_preprocessing=False,
        sample_ids=None,
    ):
        """
        Prepare mapping of Numpy arrays to PyTorch tensors.
        This method overwrites the method from the base class.
        The input data are preprocessed (data standarization).

        Parameters:
        ----------
            (....): See docstring of method
                :meth:`data.dataset.Dataset.input_to_torch_tensor`.

        Returns:
        ---------
            (torch.Tensor): The given input ``x`` as PyTorch tensor.
        """
        if not force_no_preprocessing:
            if mode == "inference":
                transform = self.test_transform
            elif mode == "train":
                transform = self.train_transform
            else:
                raise ValueError(
                    f"{mode} is not a valid value for the" "argument 'mode'."
                )
            return CUB2002011.torch_preprocess_images(x, device, transform)

        else:
            return Dataset.input_to_torch_tensor(
                self,
                x,
                device,
                mode=mode,
                force_no_preprocessing=force_no_preprocessing,
                sample_ids=sample_ids,
            )

    @staticmethod
    def torch_preprocess_images(x, device, transform, img_shape=[32, 32, 3]):
        """
        Prepare preprocessing of TinyImageNet images with a selected
        PyTorch transformation.

        Parameters:
        ----------
            x: np.ndarray)
                2D array containing TinyImageNet images.
            device: torch.device or int: 
                PyTorch device on which a final tensor will be moved.
            transform: torchvision.transforms
                A method of data modification.

        Returns:
        --------
            (torch.Tensor): The preprocessed images as PyTorch tensor.
        """
        assert len(x.shape) == 2
        # First dimension is related to batch size and second is related
        # to the flattened image.

        x = CUB2002011.load_images_to_tensor(x)
        x = torch.tensor(x * 255.0, dtype=torch.uint8)
        x = x.reshape(-1, *img_shape)
        x = torch.stack([transform(x[i, ...]) for i in range(x.shape[0])]).to(
            device
        )
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, np.prod((3,32,32)))
        return x
    
    # Function to load a batch of image paths and convert to tensors
    @staticmethod
    def load_images_to_tensor(image_paths):
        tensors = []
        transform = transforms.Compose([
            transforms.Resize((32, 32)),             
            transforms.ToTensor(), 
            ])
        for path in image_paths:
            # Load the image
            img = Image.open(path[0]).convert('RGB')  # Ensure RGB format
            img_tensor = transform(img)
            tensors.append(img_tensor)

        # Stack tensors into a single batch
        batch_tensor = torch.stack(tensors)
        return batch_tensor


class SplitCUB200Data(CUB2002011):
    """An instance of the class shall represent a SplitCUB200 task.

    Paramaters:
    ------------
        data_path: str
            Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot: bool
            Whether the class labels should be represented in a one-hot encoding.
        validation_size: int
            The number of validation samples. Validation samples will be taking from
            the training set (the first :math:`n` samples).
        use_torch_augmentation: bool
            Flag to indicate whether data augmentation should be used or not.
        labels: List
            The labels that should be part of this task.
        full_out_dim: bool
            Choose the original CUB200 instead of the new task output dimension. 
            This option will affect the attributes :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
        trgt_padding: int, optional:
            If provided, ``trgt_padding`` fake classes
            will be added, such that in total the returned dataset has
            ``len(labels) + trgt_padding`` classes. However, all padded classes
            have no input instances. Note, that 1-hot encodings are padded to
            fit the new number of classes.
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=1000,
                 use_torch_augmentation=False, labels=range(0,10),
                 full_out_dim=False, trgt_padding=None):
        # Note, we build the validation set below!
        super().__init__(data_path, use_one_hot=use_one_hot, num_val_per_class=int(validation_size // len(labels)))


        self._full_out_dim = full_out_dim

        if isinstance(labels, range):
            labels = list(labels)
        assert np.all(np.array(labels) >= 0) and \
               np.all(np.array(labels) < self.num_classes) and \
               len(labels) == len(np.unique(labels))
        K = len(labels)

        self._labels = labels

        train_ins = self.get_train_inputs()
        test_ins = self.get_test_inputs()

        train_outs = self.get_train_outputs()
        test_outs = self.get_test_outputs()

        # Get labels.
        if self.is_one_hot:
            train_labels = self._to_one_hot(train_outs, reverse=True)
            test_labels = self._to_one_hot(test_outs, reverse=True)
        else:
            train_labels = train_outs
            test_labels = test_outs

        train_labels = train_labels.squeeze()
        test_labels = test_labels.squeeze()

        train_mask = train_labels == labels[0]
        test_mask = test_labels == labels[0]
        for k in range(1, K):
            train_mask = np.logical_or(train_mask, train_labels == labels[k])
            test_mask = np.logical_or(test_mask, test_labels == labels[k])

        train_ins = train_ins[train_mask, :]
        test_ins = test_ins[test_mask, :]

        train_outs = train_outs[train_mask, :]
        test_outs = test_outs[test_mask, :]

        if validation_size > 0:
            if validation_size >= train_outs.shape[0]:
                raise ValueError('Validation set size must be smaller than ' +
                                 '%d.' % train_outs.shape[0])
            val_inds = np.arange(validation_size)
            train_inds = np.arange(validation_size, train_outs.shape[0])

        else:
            train_inds = np.arange(train_outs.shape[0])

        test_inds = np.arange(train_outs.shape[0],
                              train_outs.shape[0] + test_outs.shape[0])

        outputs = np.concatenate([train_outs, test_outs], axis=0)

        if not full_out_dim:
            # Transform outputs, e.g., if 1-hot [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            # Note, the method assumes `full_out_dim` when later called by a
            # user. We just misuse the function to call it inside the
            # constructor.
            self._full_out_dim = True
            outputs = self.transform_outputs(outputs)
            self._full_out_dim = full_out_dim

            # Note, we may also have to adapt the output shape appropriately.
            if self.is_one_hot:
                self._data['out_shape'] = [len(labels)]

        images = np.concatenate([train_ins, test_ins], axis=0)

        ### Overwrite internal data structure. Only keep desired labels.

        # Note, we continue to pretend to be a 10 class problem, such that
        # the user has easy access to the correct labels and has the original
        # 1-hot encodings.
        if not full_out_dim:
            self._data['num_classes'] = len(labels)
        else:
            self._data['num_classes'] = 10
        self._data['in_data'] = images
        self._data['out_data'] = outputs
        self._data['train_inds'] = train_inds
        self._data['test_inds'] = test_inds
        if validation_size > 0:
            self._data['val_inds'] = val_inds

        n_val = 0
        if validation_size > 0:
            n_val = val_inds.size

        if trgt_padding is not None and trgt_padding > 0:
            print('SplitCUB200 targets will be padded with %d zeroes.' \
                  % trgt_padding)
            self._data['num_classes'] += trgt_padding

            if self.is_one_hot:
                self._data['out_shape'] = [self._data['out_shape'][0] + \
                                           trgt_padding]
                out_data = self._data['out_data']
                self._data['out_data'] = np.concatenate((out_data,
                    np.zeros((out_data.shape[0], trgt_padding))), axis=1)

        print('Created SplitCUB200 task with labels %s and %d train, %d test '
              % (str(labels), train_inds.size, test_inds.size) +
              'and %d val samples.' % (n_val))


    def transform_outputs(self, outputs):
        """Transform the outputs from the 10D CUB200 dataset into proper labels
        based on the constructor argument ``labels``.

        I.e., the output will have ``len(labels)`` classes.

        Example:
            Split with labels [2,3]

            1-hot encodings: [0,0,0,1,0,0,0,0,0,0] -> [0,1]

            labels: 3 -> 1

        Paramaters:
        -----------
            outputs: np.ndarray
                2D numpy array of outputs.

        Returns:
        --------
            2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)


    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitCUB200'


if __name__ == '__main__':
    pass