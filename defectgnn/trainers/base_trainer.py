import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from defectgnn.utils.register import registers
from defectgnn.utils.common_util import LinearWarmupExponentialDecay
from abc import ABC, abstractmethod


@registers.task.register("base_trainer")
class BaseTrainer(ABC):
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        warmup_steps=None,
        decay_steps=100000,
        decay_rate=0.96,
        ema_decay=0.999,
        max_grad_norm=10.0,
        use_huber_loss=False,
        huber_delta=1.0,
        l2_loss_decay=0.0,
    ):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.l2_loss_decay = l2_loss_decay

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)


        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)
        self.active_trainable_weights = None

        # Initialize backup variables
        if model.built:
            weights = self._ema_weights()
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in weights]
        else:
            self.backup_vars = None

    def update_weights(self, loss, gradient_tape, variables=None):
        if variables is None:
            variables = self._ema_weights()
        grads = gradient_tape.gradient(loss, variables)
        grads_and_vars = [(grad, var) for grad, var in zip(grads, variables) if grad is not None]
        if not grads_and_vars:
            return
        grads, variables = zip(*grads_and_vars)
        variables = list(variables)
        grads = list(grads)
        self.active_trainable_weights = variables

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, variables))

    def regression_loss(self, targets, preds):
        if self.use_huber_loss:
            return tf.keras.losses.huber(
                targets, preds, delta=self.huber_delta)
        return tf.losses.mean_squared_error(targets, preds)

    def regularization_loss(self, variables):
        if not self.l2_loss_decay:
            return tf.constant(0.0, dtype=tf.float32)

        l2_terms = [
            tf.nn.l2_loss(var)
            for var in variables
            if len(var.shape) > 1
        ]
        if not l2_terms:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.cast(self.l2_loss_decay, tf.float32) * tf.add_n(l2_terms)

    def _ema_weights(self):
        if self.active_trainable_weights is not None:
            return self.active_trainable_weights
        return self.model.trainable_weights

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self._ema_weights())

    def save_variable_backups(self):
        weights = self._ema_weights()
        if self.backup_vars is None:
            self.backup_vars = [
                tf.Variable(var, dtype=var.dtype, trainable=False)
                for var in weights]
        else:
            for var, bck in zip(weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self._ema_weights(), self.backup_vars):
            var.assign(bck)

    @abstractmethod
    def train_on_batch(self, dataset_iter, metrics):
        """Derived classes should implement this function."""

    @abstractmethod
    def test_on_batch(self, dataset_iter, metrics):
        """Derived classes should implement this function."""

class UncertaintyWeightedLoss:

    def __init__(self,
                 num_tasks=None,
                 init_sigma=None):
        if init_sigma is None:
            init_vars = [1.0] * num_tasks
        elif isinstance(init_sigma, (float, int)):
            init_vars = [init_sigma] * num_tasks
        elif isinstance(init_sigma, list):
            if len(init_sigma) != num_tasks:
                raise ValueError(f"init_sigma length ({len(init_sigma)}) not equal to number of tasks ({num_tasks})")
            init_vars = init_sigma
        elif isinstance(init_sigma,dict):
            init_vars = [v for v in init_sigma.values()]
        else:
            raise ValueError("init sigma only support int, float, list and dict.")
        self.vars = tf.Variable(
            [np.log(sigma) for sigma in init_vars],
            trainable=True,
            dtype=tf.float32,
            name='uncertainty_weights'
        )
        self.num_tasks = num_tasks

    @property
    def log_sigmas(self):
        return self.vars

    @property
    def get_variables(self):
        return [self.vars]

