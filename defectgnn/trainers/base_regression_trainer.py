import logging
import tensorflow as tf
import numpy as np
from keras import backend as K
from defectgnn import registers
from defectgnn.trainers.base_trainer import BaseTrainer, UncertaintyWeightedLoss

def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y,y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2

def pearson_r(y_true, y_pred):
    # A significant portion of the code is adapted from Keras_Metrics (https://github.com/WenYanger/Keras_Metrics).
    epsilon = 10e-5
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return K.mean(r)

@registers.task.register("base_regression_trainer")
class BaseRegressionTrainer(BaseTrainer):
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        model,
        target_types,
        learning_rate=1e-3,
        warmup_steps=None,
        decay_steps=100000,
        decay_rate=0.96,
        ema_decay=0.999,
        max_grad_norm=10.0,
        weight_lr=None,
        alpha=None,
        use_huber_loss=False,
        huber_delta=1.0,
        l2_loss_decay=0.0,
        init_sigma=None,
        reduce_method="mean",
        **kwargs,
    ):
        super().__init__(model=model,
                         learning_rate=learning_rate,
                         warmup_steps=warmup_steps,
                         decay_steps=decay_steps,
                         decay_rate=decay_rate,
                         ema_decay=ema_decay,
                         max_grad_norm=max_grad_norm,
                         use_huber_loss=use_huber_loss,
                         huber_delta=huber_delta,
                         l2_loss_decay=l2_loss_decay,
                         )

        self.target_types = target_types

        # Initialize loss weights
        # NOTE: use task uncertainty as loss weights; also can use GradNorm?
        num_tasks = len(self.target_types)
        self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks,init_sigma=init_sigma)

        # Initialize metrics
        self.metrics_dict = dict()
        for target_type in self.target_types:
            self.metrics_dict[target_type] = {
                "mae": tf.keras.metrics.MeanAbsoluteError(),
                "mse": tf.keras.metrics.MeanSquaredError(),
                "rmse": tf.keras.metrics.RootMeanSquaredError(),
            }

        self.reduce_method = reduce_method

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        target_results = dict()
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)


            task_losses = []
            for target_type in self.target_types:
                self.metrics_dict[target_type]['mae'].update_state(targets[target_type], preds[target_type])
                self.metrics_dict[target_type]['mse'].update_state(targets[target_type], preds[target_type])
                self.metrics_dict[target_type]['rmse'].update_state(targets[target_type], preds[target_type])

                mae = self.metrics_dict[target_type]['mae'].result()
                mse = self.metrics_dict[target_type]['mse'].result()
                rmse = self.metrics_dict[target_type]['rmse'].result()

                r2 = R_squared(targets[target_type], preds[target_type])
                pearson = pearson_r(targets[target_type], preds[target_type])

                self.metrics_dict[target_type]['mae'].reset_states()
                self.metrics_dict[target_type]['mse'].reset_states()
                self.metrics_dict[target_type]['rmse'].reset_states()

                loss = self.regression_loss(targets[target_type], preds[target_type])
                if self.reduce_method == "mean":
                    loss_scaler = tf.reduce_mean(loss)
                elif self.reduce_method == "sum":
                    loss_scaler = tf.reduce_sum(loss)
                else:
                    raise ValueError("reduce method only choose from mean or sum")
                task_losses.append(loss_scaler)

                target_results[target_type] = {
                    'loss': loss,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'pearson': pearson
                }


            if len(self.target_types) == 1:
                total_loss = task_losses[0]
            else:
                total_loss = 0.0
                log_sigmas = self.uncertainty_loss.log_sigmas
                for i, loss in enumerate(task_losses):
                    weight = tf.exp(-2 * log_sigmas[i])
                    weighted_loss = weight * loss + 2 * log_sigmas[i]
                    total_loss += weighted_loss
            total_loss += self.regularization_loss(self.model.trainable_variables)

        # Update variables
        trainable_vars = self.model.trainable_variables
        if len(self.target_types) > 1:
            trainable_vars += self.uncertainty_loss.get_variables
        self.update_weights(total_loss, tape, variables=trainable_vars)

        for target_type in self.target_types:
            result_dict = target_results[target_type]
            loss = result_dict['loss']
            mae = result_dict['mae']
            mse = result_dict['mse']
            rmse = result_dict['rmse']
            r2 = result_dict['r2']
            pearson = result_dict['pearson']
            metrics[target_type].update_state(loss, mae, mse, rmse, r2, pearson, 1)
        return total_loss, targets, preds

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)


        preds = self.model(inputs, training=False)

        target_results = dict()


        task_losses = []
        for target_type in self.target_types:
            self.metrics_dict[target_type]['mae'].update_state(targets[target_type], preds[target_type])
            self.metrics_dict[target_type]['mse'].update_state(targets[target_type], preds[target_type])
            self.metrics_dict[target_type]['rmse'].update_state(targets[target_type], preds[target_type])

            mae = self.metrics_dict[target_type]['mae'].result()
            mse = self.metrics_dict[target_type]['mse'].result()
            rmse = self.metrics_dict[target_type]['rmse'].result()


            r2 = 0.0
            pearson = 0.0

            self.metrics_dict[target_type]['mae'].reset_states()
            self.metrics_dict[target_type]['mse'].reset_states()
            self.metrics_dict[target_type]['rmse'].reset_states()

            loss = self.regression_loss(targets[target_type], preds[target_type])
            if self.reduce_method == "mean":
                loss_scaler = tf.reduce_mean(loss)
            elif self.reduce_method == "sum":
                loss_scaler = tf.reduce_sum(loss)
            else:
                raise ValueError("reduce method only choose from mean or sum")
            task_losses.append(loss_scaler)

            target_results[target_type] = {
                'loss': loss,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'pearson': pearson
            }
        if len(self.target_types) == 1:
            total_loss = task_losses[0]
        else:
            total_loss = 0.0
            log_sigmas = self.uncertainty_loss.log_sigmas
            for i, loss in enumerate(task_losses):
                weight = tf.exp(-2 * log_sigmas[i])
                weighted_loss = weight * loss + 2 * log_sigmas[i]
                total_loss += weighted_loss

        for target_type in self.target_types:
            result_dict = target_results[target_type]
            loss = result_dict['loss']
            mae = result_dict['mae']
            mse = result_dict['mse']
            rmse = result_dict['rmse']
            r2 = result_dict['r2']
            pearson = result_dict['pearson']
            metrics[target_type].update_state(loss, mae, mse, rmse, r2, pearson, 1)
        return total_loss, targets, preds


