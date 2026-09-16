import tensorflow as tf
import numpy as np
from keras import backend as K
from defectgnn import registers
from defectgnn.trainers.base_trainer import BaseTrainer

def R_squared(y, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y,y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2

def pearson_r(y_true, y_pred):
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

@registers.task.register("gradnorm_trainer")
class GradNormTrainer(BaseTrainer):
    """
    GradNorm implementation adapted to the BaseTrainer structure.
    """
    def __init__(
        self,
        model,
        target_types,
        alpha=1.5,
        learning_rate=1e-3,
        weight_lr=0.001,
        warmup_steps=None,
        decay_steps=100000,
        decay_rate=0.96,
        ema_decay=0.999,
        max_grad_norm=10.0,
        use_huber_loss=False,
        huber_delta=1.0,
        l2_loss_decay=0.0,
        reduce_method="mean",
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
        self.reduce_method = reduce_method
        self.alpha = alpha

        num_tasks = len(self.target_types)

        # --- GradNorm Specific Init ---
        self.task_weights = tf.Variable(
            tf.ones(num_tasks, dtype=tf.float32),
            trainable=True,
            name="gradnorm_weights"
        )

        self.initial_losses = tf.Variable(
            tf.zeros(num_tasks, dtype=tf.float32),
            trainable=False,
            name="initial_losses"
        )
        self.is_first_step = tf.Variable(True, trainable=False)

        self.weight_optimizer = tf.optimizers.Adam(learning_rate=weight_lr)
        # ------------------------------

        self.metrics_dict = dict()
        for target_type in self.target_types:
            self.metrics_dict[target_type] = {
                "mae": tf.keras.metrics.MeanAbsoluteError(),
                "mse": tf.keras.metrics.MeanSquaredError(),
                "rmse": tf.keras.metrics.RootMeanSquaredError(),
            }

    def _get_shared_layer_weights(self):
        """
        Helper to get the weights of the last shared layer from the GNN model.
        Requires the model to implement `get_last_shared_layer_params`.
        """
        if hasattr(self.model, 'get_last_shared_layer_params'):
            return self.model.get_last_shared_layer_params()
        else:
            try:
                return self.model.blocks[-1].node_mlp.kernel
            except:
                raise NotImplementedError(
                    "GradNorm requires `model.get_last_shared_layer_params()` to be implemented."
                )

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        target_results = dict()


        is_single_task = (len(self.target_types) == 1)

        with tf.GradientTape(persistent=True) as tape:
            preds = self.model(inputs, training=True)

            task_losses = []

            for i, target_type in enumerate(self.target_types):
                # Update Metrics
                self.metrics_dict[target_type]['mae'].update_state(targets[target_type], preds[target_type])
                self.metrics_dict[target_type]['mse'].update_state(targets[target_type], preds[target_type])
                self.metrics_dict[target_type]['rmse'].update_state(targets[target_type], preds[target_type])

                mae = self.metrics_dict[target_type]['mae'].result()
                mse = self.metrics_dict[target_type]['mse'].result()
                rmse = self.metrics_dict[target_type]['rmse'].result()

                r2 = R_squared(targets[target_type], preds[target_type])
                pearson = pearson_r(targets[target_type], preds[target_type])

                # Reset states for next batch
                self.metrics_dict[target_type]['mae'].reset_states()
                self.metrics_dict[target_type]['mse'].reset_states()
                self.metrics_dict[target_type]['rmse'].reset_states()

                # Compute Raw Loss
                loss = self.regression_loss(targets[target_type], preds[target_type])
                if self.reduce_method == "mean":
                    loss_scaler = tf.reduce_mean(loss)
                elif self.reduce_method == "sum":
                    loss_scaler = tf.reduce_sum(loss)
                else:
                    raise ValueError("reduce method only choose from mean or sum")

                task_losses.append(loss_scaler)

                target_results[target_type] = {
                    'loss': loss_scaler,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'pearson': pearson,
                    'weight': self.task_weights[i]
                }

            # Stack losses for vectorized operations
            loss_vec = tf.stack(task_losses)
            model_trainable_variables = [
                var for var in tape.watched_variables()
                if var is not self.task_weights
            ]

            # Loss_total = sum(w_i * L_i)
            if is_single_task:
                total_loss = loss_vec[0]
                grad_norm_loss = 0.0
            else:
                if self.is_first_step:
                    self.initial_losses.assign(loss_vec)
                    self.is_first_step.assign(False)
                total_loss = tf.reduce_sum(self.task_weights * loss_vec)
            total_loss += self.regularization_loss(model_trainable_variables)

            # ---------------- GradNorm Logic Start ----------------
            if not is_single_task:
                shared_weights = self._get_shared_layer_weights()

                norms = []
                for i in range(len(self.target_types)):
                    grad = tape.gradient(self.task_weights[i] * loss_vec[i], shared_weights)
                    if grad is None:
                        tf.print(f"Warning: Gradient for task {i} on shared layer is None!")
                        grad_norm = tf.constant(0.0)
                    else:
                        grad_norm = tf.norm(grad)
                    norms.append(tf.norm(grad_norm))
                norms = tf.stack(norms)  # [G_1, G_2, ...]

                mean_norm = tf.reduce_mean(norms)

                loss_ratios = loss_vec / (self.initial_losses + 1e-8)
                inverse_train_rates = loss_ratios / tf.reduce_mean(loss_ratios)

                target_norms = mean_norm * (inverse_train_rates ** self.alpha)
                target_norms = tf.stop_gradient(target_norms)

                # GradNorm Loss: L_grad = sum |G_i - G_target|_1
                grad_norm_loss = tf.reduce_sum(tf.abs(norms - target_norms))
                # ---------------- GradNorm Logic End ----------------

        # --- Update Step ---

        self.update_weights(total_loss, tape, variables=model_trainable_variables)

        if not is_single_task:
            weight_grads = tape.gradient(grad_norm_loss, self.task_weights)
            self.weight_optimizer.apply_gradients(zip([weight_grads], [self.task_weights]))

            num_tasks_float = tf.cast(len(self.target_types), tf.float32)
            new_weights = self.task_weights / tf.reduce_sum(self.task_weights) * num_tasks_float
            self.task_weights.assign(new_weights)

        del tape

        for target_type in self.target_types:
            result_dict = target_results[target_type]
            metrics[target_type].update_state(
                result_dict['loss'],
                result_dict['mae'],
                result_dict['mse'],
                result_dict['rmse'],
                result_dict['r2'],
                result_dict['pearson'],
                1
            )

        if not is_single_task:
            debug_info = {
                "w": self.task_weights,
                "g_norm": norms,
                "g_target": target_norms,
                "l_ratio": loss_ratios,
                "grad_loss": grad_norm_loss
            }

        return {
            "total_loss": total_loss,
            "targets": targets,
            "preds": preds,
            "debug_info": debug_info
        } if not is_single_task else {
            "total_loss": total_loss,
        }

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

            task_losses.append(loss_scaler)

            target_results[target_type] = {
                'loss': loss_scaler,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'pearson': pearson
            }

        loss_vec = tf.stack(task_losses)
        total_loss = tf.reduce_sum(self.task_weights * loss_vec)

        for target_type in self.target_types:
            result_dict = target_results[target_type]
            metrics[target_type].update_state(
                result_dict['loss'],
                result_dict['mae'],
                result_dict['mse'],
                result_dict['rmse'],
                result_dict['r2'],
                result_dict['pearson'],
                1
            )
        return total_loss, targets, preds

