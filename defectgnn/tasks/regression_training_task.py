import os
import pickle
import logging
import string
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from defectgnn.tasks.base_training_task import BaseTrainingTask
from defectgnn import swish, registers, RegressionMetrics


def r2_score_numpy(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.array(y_pred, dtype=np.float32).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < eps:
        return 0.0
    return 1.0 - ss_res / ss_tot

def pearson_r_numpy(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.array(y_pred, dtype=np.float32).reshape(-1)
    xm = y_true - y_true.mean()
    ym = y_pred - y_pred.mean()
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if r_den < eps:
        return 0.0
    return float(r_num / r_den)

@registers.task.register("regression_training_task")
class RegressionTrainingTask(BaseTrainingTask):

    def __init__(
            self,
            args
    ):
        super().__init__(args=args)



        self.is_debug = getattr(self.args.misc, 'debug', False)
        self.debug_step_freq = getattr(self.args.misc, 'debug_step', 100)

    def initial_metrics(self):
        self.training_metrics = dict()
        self.validation_metrics = dict()
        self.test_metrics = dict()
        for target_type in self.args.data.target_types:

            self.training_metrics[target_type] = RegressionMetrics('train', ["targets"])
            self.validation_metrics[target_type] = RegressionMetrics('validation', ["targets"])
            self.test_metrics[target_type] = RegressionMetrics('test', ["targets"])

    def _count_params(self, variables):
        return int(np.sum([np.prod(var.shape.as_list()) for var in variables]))

    def _log_trainable_summary(self, freeze_policy):
        try:
            trainable_variables = self.model.trainable_variables
        except ValueError as exc:
            logging.info(
                f"Freeze policy: {freeze_policy}. Trainable variable count "
                f"will be available after the first model call. {exc}")
            return

        logging.info(
            f"Freeze policy: {freeze_policy}. Trainable variables: "
            f"{len(trainable_variables)}, params: {self._count_params(trainable_variables)}")
        for var in trainable_variables:
            logging.info(f"Trainable after freeze: {var.name} shape={var.shape}")

    def _freeze_all_model_parts(self):
        self.model.trainable = True
        for layer_name in ("embedding_layer", "feat_integrate_block"):
            if hasattr(self.model, layer_name):
                getattr(self.model, layer_name).trainable = False
        for block_list_name in ("bond2bond_blocks", "bond2atom_blocks", "atom2bond_blocks"):
            for block in getattr(self.model, block_list_name, []):
                block.trainable = False
        for readout in getattr(self.model, "readout_blocks", {}).values():
            readout.trainable = False

    def apply_freeze_policy(self):
        freeze_policy = getattr(self.args.misc, "freeze_policy", None)
        if freeze_policy is None or freeze_policy == "none":
            logging.info("Freeze policy: none. All model variables remain trainable.")
            return

        if freeze_policy not in {
                "atom_readout_only",
                "atom_head",
                "path_readout_only",
                "readouts_only",
                "freeze_atom_readout",
                "freeze_path_readout"}:
            raise ValueError(
                "misc.freeze_policy must be one of: none, atom_readout_only, "
                "atom_head, path_readout_only, readouts_only, "
                "freeze_atom_readout, freeze_path_readout")

        if freeze_policy in {"freeze_atom_readout", "freeze_path_readout"}:
            self.model.trainable = True
            target_type = "atom" if freeze_policy == "freeze_atom_readout" else "path"
            if target_type not in self.model.readout_blocks:
                raise ValueError(
                    f"freeze_policy={freeze_policy} requires {target_type} in target_types.")
            self.model.readout_blocks[target_type].trainable = False
            if hasattr(self.trainer, "backup_vars"):
                self.trainer.backup_vars = None

            logging.info(f"Frozen readout: {target_type}")
            self._log_trainable_summary(freeze_policy)
            return

        self._freeze_all_model_parts()

        if freeze_policy in {"atom_readout_only", "atom_head", "readouts_only"}:
            if "atom" not in self.model.readout_blocks:
                raise ValueError("freeze_policy requires an atom readout, but target_types does not include atom.")
            self.model.readout_blocks["atom"].trainable = True

        if freeze_policy == "atom_head":
            if hasattr(self.model, "feat_integrate_block"):
                self.model.feat_integrate_block.trainable = True

        if freeze_policy in {"path_readout_only", "readouts_only"}:
            if "path" not in self.model.readout_blocks:
                raise ValueError("freeze_policy requires a path readout, but target_types does not include path.")
            self.model.readout_blocks["path"].trainable = True

        if hasattr(self.trainer, "backup_vars"):
            self.trainer.backup_vars = None

        self._log_trainable_summary(freeze_policy)

    def run(self):
        num_train, num_validation, num_test = self.data_provider.get_train_validation_test_num()



        best_composite_score = -float('inf')
        best_epoch = 0

        # NOTE:read exist data or initialize setting
        if os.path.isfile(self.best_loss_file):
            try:
                loss_file = np.load(self.best_loss_file, allow_pickle=True)

                metrics_best = {k: v.item() if not isinstance(v, np.ndarray) else v for k, v in loss_file.items()}

                if 'composite_score' in metrics_best:
                    best_composite_score = metrics_best['composite_score']
                    best_epoch = metrics_best['best_epoch']
            except Exception as e:
                logging.warning(f"Failed to load best_loss_file: {e}. Starting fresh.")
                metrics_best = {}
        else:
            metrics_best = {}

            # for target_type in self.args.data.target_types:
            #     metrics_best[target_type] = self.validation_metrics[target_type].result()
            #     for key in metrics_best[target_type].keys():
            #         if key == "mean_r2_validation":
            #             metrics_best[target_type][key] = -1000
            #         else:
            #             metrics_best[target_type][key] = 0
            #     metrics_best[target_type]['step'] = 0
            #     metrics_best[target_type]['epoch'] = 0
            # np.savez(self.best_loss_file, **metrics_best)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.trainer.optimizer, model=self.model)

        if hasattr(self.trainer, 'task_weights'):
             ckpt.task_weights = self.trainer.task_weights
        if hasattr(self.trainer, 'weight_optimizer'):
             ckpt.weight_optimizer = self.trainer.weight_optimizer

        manager = tf.train.CheckpointManager(ckpt, self.logger_path, max_to_keep=self.args.misc.ckpt_max_to_keep)

        # Restore latest checkpoint. By default resume the current run; for
        # fine-tuning, misc.restore_ckpt_path can point to an existing logs dir
        # or to a model directory containing logs/.
        restore_ckpt_path = getattr(self.args.misc, "restore_ckpt_path", None)
        if restore_ckpt_path is not None:
            restore_ckpt_path = os.path.expanduser(restore_ckpt_path)
            if os.path.isdir(os.path.join(restore_ckpt_path, "logs")):
                restore_ckpt_path = os.path.join(restore_ckpt_path, "logs")
        else:
            restore_ckpt_path = self.logger_path
        ckpt_restored = tf.train.latest_checkpoint(restore_ckpt_path)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored).expect_partial()
            logging.info(f"Restored from checkpoint: {ckpt_restored}")

        self.apply_freeze_policy()

        # Save normalizer & args (Standard)
        if self.data_container.target_normalizer is not None:
            with open(os.path.join(self.best_path, "normalizer.pkl"), 'wb') as f:
                pickle.dump(self.data_container.target_normalizer, f, protocol=4)
        with open(os.path.join(self.best_path, 'args.pkl'), "wb") as f:
            pickle.dump({"args": vars(self.args)}, f, protocol=4)

        with self.summary_writer.as_default():
            steps_per_epoch = int(np.ceil(num_train / self.args.training.batch_size))
            step_initial = int(ckpt.step.numpy()) + 1 if ckpt_restored is not None else 1
            first_save_model = True

            for step in range(step_initial, self.args.training.num_steps + 1):
                # Update step number
                ckpt.step.assign(step)
                tf.summary.experimental.set_step(step)



                train_results = self.trainer.train_on_batch(self.train_dataset, self.training_metrics)


                if isinstance(train_results, dict):
                    total_loss = train_results["total_loss"]


                    debug_data = train_results.get("debug_info", None)
                else:

                    total_loss, _, _ = train_results
                    debug_data = None


                if self.is_debug and debug_data is not None and step % self.debug_step_freq == 0:
                    self._log_debug_info(step, debug_data)


                if isinstance(train_results, dict) and 'w_ef' in train_results:

                    for i, t_type in enumerate(self.args.data.target_types):
                        if hasattr(self.trainer, 'task_weights'):
                            tf.summary.scalar(f"GradNorm_Weights/{t_type}", self.trainer.task_weights[i])

                # Validation Phase
                if step % steps_per_epoch == 0:
                    epoch = step // steps_per_epoch
                    manager.save()

                    # EMA handling
                    self.trainer.save_variable_backups()
                    self.trainer.load_averaged_variables()

                    # Data Containers
                    validation_targets_dict = defaultdict(list)
                    validation_preds_dict = defaultdict(list)
                    test_targets_dict = defaultdict(list)
                    test_preds_dict = defaultdict(list)

                    # --- Validation Loop ---

                    for _ in range(int(np.ceil(num_validation / self.args.training.batch_size))):
                        _, val_targets, val_preds = self.trainer.test_on_batch(
                            self.validation_dataset, self.validation_metrics)

                        for target_type in self.args.data.target_types:

                            t_data = val_targets[target_type].numpy().reshape(-1, 1)
                            p_data = val_preds[target_type].numpy().reshape(-1, 1)


                            if self.data_container.target_normalizer is not None:
                                normalizer = self.data_container.target_normalizer[target_type]
                                t_data = normalizer.inverse_transform(t_data)
                                p_data = normalizer.inverse_transform(p_data)

                            validation_targets_dict[target_type].extend(t_data.reshape(-1).tolist())
                            validation_preds_dict[target_type].extend(p_data.reshape(-1).tolist())

                    # --- Test Loop ---


                    for _ in range(int(np.ceil(num_test / self.args.training.batch_size))):
                        _, t_targets, t_preds = self.trainer.test_on_batch(
                            self.test_dataset, self.test_metrics)

                        for target_type in self.args.data.target_types:
                            t_data = t_targets[target_type].numpy().reshape(-1, 1)
                            p_data = t_preds[target_type].numpy().reshape(-1, 1)

                            if self.data_container.target_normalizer is not None:
                                normalizer = self.data_container.target_normalizer[target_type]
                                t_data = normalizer.inverse_transform(t_data)
                                p_data = normalizer.inverse_transform(p_data)

                            test_targets_dict[target_type].extend(t_data.reshape(-1).tolist())
                            test_preds_dict[target_type].extend(p_data.reshape(-1).tolist())

                    # TODO: calculate the metrics and save the best model parameters

                    # --- Metrics Calculation ---
                    current_epoch_r2_sum = 0.0

                    for target_type in self.args.data.target_types:
                        # Validation Metrics
                        val_r2 = r2_score_numpy(validation_targets_dict[target_type],
                                                validation_preds_dict[target_type])
                        val_pearson = pearson_r_numpy(validation_targets_dict[target_type],
                                                      validation_preds_dict[target_type])

                        # Test Metrics
                        test_r2 = r2_score_numpy(test_targets_dict[target_type], test_preds_dict[target_type])
                        test_pearson = pearson_r_numpy(test_targets_dict[target_type], test_preds_dict[target_type])

                        # Store in Metrics Object
                        self.validation_metrics[target_type].mean_r2 = val_r2
                        self.validation_metrics[target_type].mean_pearson = val_pearson
                        self.test_metrics[target_type].mean_r2 = test_r2
                        self.test_metrics[target_type].mean_pearson = test_pearson


                        current_epoch_r2_sum += val_r2


                    current_composite_score = current_epoch_r2_sum / len(self.args.data.target_types)

                    # Log TensorBoard for Composite Score
                    tf.summary.scalar('Composite/Validation_Mean_R2', current_composite_score)

                    # --- Model Saving Logic ---
                    is_improved = False


                    if current_composite_score > best_composite_score:
                        is_improved = True
                        best_composite_score = current_composite_score
                        best_epoch = epoch


                        metrics_best['composite_score'] = best_composite_score
                        metrics_best['best_epoch'] = best_epoch
                        for target_type in self.args.data.target_types:
                            metrics_best[target_type] = self.validation_metrics[target_type].result()
                            metrics_best[target_type]['epoch'] = epoch

                    if is_improved:
                        logging.info(
                            f"*** New Best Model (Epoch {epoch})! Composite R2: {best_composite_score:.4f} ***")
                        np.savez(self.best_loss_file, **metrics_best)

                        # Save Model Structure (Once)
                        if first_save_model and not getattr(self.args.misc, 'skip_full_model_save', False):
                            self.model.save(os.path.join(self.best_path, f'best-full-model-epoch{epoch}'))
                            with open(os.path.join(self.best_path, 'best_logger.txt'), "a") as f:
                                f.write(f"Model Summary Saved.\n")
                            first_save_model = False
                        elif first_save_model:
                            with open(os.path.join(self.best_path, 'best_logger.txt'), "a") as f:
                                f.write("Model Summary Skipped.\n")
                            first_save_model = False

                        # Save Weights
                        if self.args.misc.save_all_models:
                            self.model.save_weights(os.path.join(self.best_path, f'best-model-epoch{epoch}'))
                        else:
                            self.model.save_weights(os.path.join(self.best_path, 'best-model'))

                        # Save Predictions (CSV)
                        self._save_predictions(epoch, validation_targets_dict, validation_preds_dict,
                                               test_targets_dict, test_preds_dict)

                        # Save Log Text
                        self._write_log_text(epoch, step)

                        # Save CSV Scores
                        self._write_csv_scores(epoch, step)

                    else:
                        # --- Early Stopping Logic ---
                        if epoch > best_epoch + self.args.training.early_stopping_epochs:
                            logging.info(
                                f"Early Stopping triggered! No improvement for {self.args.training.early_stopping_epochs} epochs. Best Epoch: {best_epoch}, Best R2: {best_composite_score:.4f}")
                            break

                    # Reset Metrics States
                    for target_type in self.args.data.target_types:
                        self.training_metrics[target_type].reset_states()
                        self.validation_metrics[target_type].reset_states()
                        self.test_metrics[target_type].reset_states()

                    # Restore EMA backups
                    self.trainer.restore_variable_backups()

    def _save_predictions(self, epoch, val_t, val_p, test_t, test_p):
        suffix = f"_epoch{epoch}" if self.args.misc.save_all_predictions else ""


        for target_type in self.args.data.target_types:



            if target_type in val_t and len(val_t[target_type]) > 0:
                try:
                    df_val = pd.DataFrame({
                        "target": val_t[target_type],
                        "pred": val_p[target_type]
                    })

                    filename = f'best_predict_validation_{target_type}{suffix}.csv'
                    df_val.to_csv(os.path.join(self.best_path, filename), index=False)
                except ValueError as e:
                    logging.error(f"Error saving validation CSV for {target_type}: {e}")


            if target_type in test_t and len(test_t[target_type]) > 0:
                try:
                    df_test = pd.DataFrame({
                        "target": test_t[target_type],
                        "pred": test_p[target_type]
                    })
                    filename = f'best_predict_test_{target_type}{suffix}.csv'
                    df_test.to_csv(os.path.join(self.best_path, filename), index=False)
                except ValueError as e:
                    logging.error(f"Error saving test CSV for {target_type}: {e}")

    def _write_log_text(self, epoch, step):
        with open(os.path.join(self.best_path, 'best_logger.txt'), "a") as file:
            file.write(f"--- Epoch {epoch} (Step {step}) ---\n")
            for target_type in self.args.data.target_types:
                file.write(
                    f"[{target_type}] "
                    f"Loss: {self.training_metrics[target_type].loss:.4f}/{self.validation_metrics[target_type].loss:.4f} (T/V) | "
                    f"MAE: {self.validation_metrics[target_type].mean_mae:.4f} | "
                    f"R2: {self.validation_metrics[target_type].mean_r2:.4f}\n"
                )

    def _write_csv_scores(self, epoch, step):
        header = "steps,epoch"
        metrics = ["loss", "mae", "mse", "rmse", "r2", "pearson"]

        for metric in metrics:
            for target_type in self.args.data.target_types:
                header += f",{target_type}_train_{metric},{target_type}_val_{metric},{target_type}_test_{metric}"

        file_path = os.path.join(self.best_path, 'best_scores.csv')

        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, "w") as f:
                f.write(header + "\n")

        line = f"{step},{epoch}"
        for metric in metrics:
            for target_type in self.args.data.target_types:

                attr_name = f'mean_{metric}' if metric != 'loss' else metric

                line += f",{getattr(self.training_metrics[target_type], attr_name):.6f}"
                line += f",{getattr(self.validation_metrics[target_type], attr_name):.6f}"
                line += f",{getattr(self.test_metrics[target_type], attr_name):.6f}"

        with open(file_path, "a") as f:
            f.write(line + "\n")


    def _log_debug_info(self, step, data):
        """Convert Tensor values to NumPy arrays and log debug details."""
        try:

            w = data['w'].numpy()
            g_norm = data['g_norm'].numpy()
            g_target = data['g_target'].numpy()
            l_ratio = data['l_ratio'].numpy()
            grad_loss = data['grad_loss'].numpy()



            types = self.args.data.target_types

            log_str = f"\n[DEBUG Step {step}] GradNorm Status:\n"
            log_str += f"  > Weights (w_i)   : " + ", ".join([f"{t}={v:.4f}" for t, v in zip(types, w)]) + "\n"
            log_str += f"  > Grad Norm (G_i) : " + ", ".join([f"{t}={v:.4f}" for t, v in zip(types, g_norm)]) + "\n"
            log_str += f"  > Target G (Target): " + ", ".join([f"{t}={v:.4f}" for t, v in zip(types, g_target)]) + "\n"
            log_str += f"  > Loss Ratio (r_i): " + ", ".join([f"{t}={v:.4f}" for t, v in zip(types, l_ratio)]) + "\n"
            log_str += f"  > GradNorm Loss   : {grad_loss:.6f}"

            logging.info(log_str)

        except Exception as e:
            logging.warning(f"Failed to log debug info: {e}")

