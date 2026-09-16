import argparse
import logging
import importlib
import tensorflow as tf
import yaml
from pathlib import Path
from defectgnn.utils.register import registers
from types import SimpleNamespace


class CommonArgs():
    def __init__(self):
        self.parser = argparse.ArgumentParser('DefectGNN')
        self.add_config_arg()

    def add_config_arg(self):
        self.parser.add_argument(
            '--config',
            type=str,
            default='train.yaml',
            help='Path to YAML configuration file'
        )

    def load_yaml_config(self, file_path: str) -> dict:
        """Load a YAML configuration file with the safe loader."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:

                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.error(f"Config file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"YAML parsing error: {e}")
            raise

    def dict_to_namespace(self, config_dict: dict) -> SimpleNamespace:
        """Recursively convert dictionaries to namespace objects."""
        if not isinstance(config_dict, dict):
            return config_dict

        for key, value in config_dict.items():
            if isinstance(value, dict):
                config_dict[key] = self.dict_to_namespace(value)
            elif isinstance(value, list):
                config_dict[key] = [
                    self.dict_to_namespace(item)
                    if isinstance(item, dict) else item
                    for item in value
                ]
        return SimpleNamespace(**config_dict)

    def get_args(self, config_file):

        config_dict = self.load_yaml_config(config_file)


        self.args = self.dict_to_namespace(config_dict)
        logging.info(f"Loaded configuration")
        return self.args

def _import_local_file(path: Path, *, project_root: Path) -> None:
    """
    Imports a Python file as a module

    :param path: The path to the file to import
    :type path: Path
    :param project_root: The root directory of the project (i.e., the "ocp" folder)
    :type project_root: Path
    """

    path = path.resolve()
    project_root = project_root.resolve()

    module_name = ".".join(
        path.absolute()
        .relative_to(project_root.absolute())
        .with_suffix("")
        .parts
    )
    logging.debug(f"Resolved module name of {path} to {module_name}")
    importlib.import_module(module_name)


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L134.
def setup_imports():
    # First, check if imports are already setup
    if registers.already_setup:
        return

    try:
        project_root = Path(__file__).resolve().absolute().parent.parent.parent
        logging.info(f"Project root: {project_root}")

        import_keys = ["data", "models", "tasks", "trainers"]
        for key in import_keys:
            for f in (project_root / "defectgnn" / key).rglob("*.py"):
                _import_local_file(f, project_root=project_root)
    finally:
        registers.already_setup = True


class LinearWarmupExponentialDecay(tf.optimizers.schedules.LearningRateSchedule):
    """
    This code is extracted from dimenet (https://github.com/gasteigerjo/dimenet).
    
    This schedule combines a linear warmup with an exponential decay.
    """
    def __init__(self, learning_rate, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.warmup = tf.optimizers.schedules.PolynomialDecay(
            1 / warmup_steps, warmup_steps, end_learning_rate=1)
        self.decay = tf.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, decay_rate)
        self.learning_rate = learning_rate

    def __call__(self, step):
        self.learning_rate = self.warmup(step) * self.decay(step)
        return self.learning_rate

    def get_learning_rate(self):
        return self.learning_rate

