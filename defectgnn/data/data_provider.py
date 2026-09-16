import random
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from defectgnn import registers
import logging

@registers.data_provider.register("data_provider")
class DataProvider:
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        data_container,
        train,
        validation,
        test,
        batch_size=1,
        random_seed=None,
        shuffle=False,
        split_level="graph",
        logging=None,
    ):
        self.data_container = data_container
        if split_level not in ("graph", "vacancy"):
            raise ValueError("split_level must be either 'graph' or 'vacancy'")
        self.split_level = split_level
        if split_level == "vacancy" and getattr(data_container, "raw_structure_list", None) is not None:
            self.n_data = len(data_container.graph_data_list)
        else:
            self.n_data = len(data_container) # NOTE:if raw_structure_list is not none, it means the number of original complete graph
        self.batch_size = batch_size
        self.target_types = self.data_container.target_types

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=random_seed)
        self.logging = logging

        if isinstance(random_seed, int):
            random.seed(random_seed)
            tf.random.set_seed(random_seed)

        # dataset split
        (train_indices, validation_indices, test_indices), (self.n_train, self.n_validation, self.n_test) = \
            get_train_validation_test_indices(
                total_size=self.n_data,
                train=train,
                validation=validation,
                test=test,
                shuffle=shuffle,
                data_container=data_container,
                split_level=split_level)

        logging.info("number of training files: {}\n".format(self.n_train))
        logging.info("training indices: {}\n".format(train_indices))

        logging.info("number of validation files: {}\n".format(self.n_validation))
        logging.info("validation indices: {}".format(validation_indices))

        logging.info("number of test files: {}\n".format(self.n_test))
        logging.info("test indices: {}".format(test_indices))

        # Store indices of training, validation and test data
        self.idx = {'train': train_indices,
                    'validation': validation_indices,
                    'test': test_indices}

        self.nsamples = {'train': self.n_train, 'validation': self.n_validation, 'test': self.n_test}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'validation': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()

        for int_key in self.data_container.int_keys():
            self.dtypes_input[int_key] = tf.int32

        for float_key in self.data_container.float_keys():
            self.dtypes_input[float_key] = tf.float32

        for number_key in self.data_container.int_number_keys():
            self.dtypes_input[number_key] = tf.int32


        self.dtypes_input['atom_features_list'] = tf.float32
        self.dtypes_input['vacancy_features'] = tf.float32


        self.dtypes_target = OrderedDict()
        for target_type in self.data_container.target_types:
            self.dtypes_target[target_type] = tf.float32

        # Shapes of dataset values
        self.shapes_input = dict()

        for int_key in self.data_container.int_keys():
            self.shapes_input[int_key] = [None]
        for float_key in self.data_container.float_keys():
            if float_key == "path_query_distances":
                self.shapes_input[float_key] = [None, 1]
            elif float_key == "atom_center_query_features":
                self.shapes_input[float_key] = [
                    None, data_container.atom_feature_len]
            else:
                self.shapes_input[float_key] = [None]
        for number_key in self.data_container.int_number_keys():
            self.shapes_input[number_key] = None


        self.shapes_input['atom_features_list'] = [None, data_container.atom_feature_len]
        self.shapes_input['vacancy_features'] = [None, data_container.atom_feature_len]

        self.shapes_target = OrderedDict()
        for target_type in self.data_container.target_types:
            self.shapes_target[target_type] = [None, 1]
            self.dtypes_input[f'reduce_to_target_indices_{target_type}'] = tf.int32
            self.shapes_input[f'reduce_to_target_indices_{target_type}'] = [None]

    def get_train_validation_test_num(self):
        """get_train_validation_test_num"""
        return self.n_train, self.n_validation, self.n_test

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                if key.startswith("reduce_to_target_indices_"):
                    target_type = key.split("_")[-1]
                    inputs_targets.append(tf.constant(batch['reduce_to_target'][target_type], dtype=dtype))
                else:
                    inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            return inputs_targets
        else:
            inputs = dict()
            for key, dtype in self.dtypes_input.items():
                if key.startswith("reduce_to_target_indices_"):
                    target_type = key.split("_")[-1]
                    inputs[f'reduce_to_target_indices_{target_type}'] = tf.constant(
                        batch["reduce_to_target_indices"][target_type], dtype=tf.int32)
                else:
                    inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = {}
            for target_type in self.target_types:
                targets[target_type] = tf.constant(batch['targets'][target_type], dtype=tf.float32)
            return (inputs, targets)

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""

        def generator():
            while True:
                idx = self.get_batch_idx(split)
                inputs, targets = self.idx_to_data(idx)
                yield inputs, targets

        return tf.data.Dataset.from_generator(
            generator,
            output_types=(dict(self.dtypes_input), dict(self.dtypes_target)),
            output_shapes=(self.shapes_input, self.shapes_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""

        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)

        return tf.data.Dataset.from_generator(
            generator,
            output_types=tf.int32,
            output_shapes=[None])

    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        inputs, targets = tf.py_function(func=lambda idx: self.idx_to_data(idx.numpy()),
                                         inp=[idx],
                                         Tout=(dict(self.dtypes_input), dict(self.dtypes_target)))


        for key, shape in self.shapes_input.items():
            inputs[key].set_shape(shape)

        for target_type, shape in self.shapes_target.items():
            targets[target_type].set_shape(shape)

        return (inputs, targets)

def get_train_validation_test_indices(
        total_size,
        train=0.8,
        validation=0.1,
        test=0.1,
        shuffle=True,
        data_container=None,
        split_level="graph",
):
    if all((train.endswith("npy"), validation.endswith("npy"), test.endswith("npy"))):
        train_indices, validation_indices, test_indices = np.load(train), np.load(validation), np.load(test)
        train_size, validation_size, test_size = len(train_indices), len(validation_indices), len(test_indices)

    else:
        train, validation, test = eval(train), eval(validation), eval(test)

        if all((isinstance(train, (list, tuple)),
                isinstance(validation, (list, tuple)),
                isinstance(test, (list, tuple)))):
            train_indices, validation_indices, test_indices = train, validation, test
            train_size, validation_size, test_size = len(train), len(validation), len(test)

        else:
            if all((isinstance(train, float),
                    isinstance(validation, float),
                    isinstance(test, float))):
                assert train + validation + test <= 1
                validation_size = int(np.ceil(validation * total_size))
                test_size = int(np.ceil(test * total_size))
                train_size = min(int(np.ceil(train * total_size)), total_size - validation_size - test_size)
            elif all((isinstance(train, int),
                      isinstance(validation, int),
                      isinstance(test, int))):
                assert train + validation + test <= total_size
                train_size, validation_size, test_size = train, validation, test
            else:
                raise ValueError("Please provide train/validation/test_size as float, int or list-like array")

            indices = list(range(total_size))
            if shuffle:
                indices = random.sample(indices, total_size)

            logging.info(f"Train size: {train_size}, Validation size: {validation_size}, Test size: {test_size}")

            if test_size != 0:
                train_indices = indices[:train_size]
                validation_indices = indices[-(validation_size + test_size):-test_size]
                test_indices = indices[-test_size:]
            else:
                train_indices = indices[:train_size]
                validation_indices = indices[-validation_size:]
                test_indices = []

    if data_container is not None and data_container.raw_structure_list is not None and split_level == "graph":

        train_ids = [i for i, sid in enumerate(data_container.raw_structure_list) if sid in train_indices]
        validation_ids = [i for i, sid in enumerate(data_container.raw_structure_list) if sid in validation_indices]
        test_ids = [i for i, sid in enumerate(data_container.raw_structure_list) if sid in test_indices]

        train_indices, validation_indices, test_indices = train_ids, validation_ids, test_ids

        train_size = len(train_indices)
        validation_size = len(validation_indices)
        test_size = len(test_indices)

    return (train_indices, validation_indices, test_indices), (train_size, validation_size, test_size)


