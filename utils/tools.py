import json
import os
import random
import pickle

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from typing import Any, IO
import sys


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def print_args(args):
    print("[INFO] Input arguments:")
    for key, val in args.items():
        print(f"[INFO]   {key}: {val}")
        

def set_random_seed(seed):
    """Sets random seed for training reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json',):
            return json.load(f)
        else:
            return ''.join(f.readlines())


def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def count_param_numbers(model):
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    return model_params


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content


def print_action_errors(action_names, errors_p1, errors_p2, acceleration_errors, joint_errors):
    """
    Prints a table of errors for 15 actions including MPJPE, P-MPJPE, and Acceleration Error.
    """
    print("+----------------+---------------------+---------------------+----------------------+")
    print("| Action         |     P1 Error(mm)    |     P2 Error(mm)    | Accel. Error(mm/s^2) |")
    print("+----------------+---------------------+---------------------+----------------------+")

    for i, action in enumerate(action_names):
        print(
            "| {0:13} | {1:20} | {2:20} | {3:21} |".format(action, errors_p1[i], errors_p2[i], acceleration_errors[i]))

    p1 = np.mean(np.array(errors_p1))
    assert round(p1, 4) == round(np.mean(joint_errors),
                                 4), f"MPJPE {p1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(acceleration_errors))
    p2 = np.mean(np.array(errors_p2))

    print("+----------------+---------------------+---------------------+----------------------+")
    print("| Average Errors | {0:20} | {1:20} | {2:21} |".format(p1, p2, acceleration_error))
    print("+----------------+---------------------+---------------------+----------------------+")
    return p1, p2, acceleration_error

def print_action_errors_mpjve(action_names, errors_p1, errors_p2, acceleration_errors, joint_errors, errors_mpjve):
    """
    Prints a table of errors for 15 actions including MPJPE, P-MPJPE, Acceleration Error, and MPJVE.
    """
    print("+----------------+---------------------+---------------------+----------------------+---------------------+")
    print("| Action         |     P1 Error(mm)    |     P2 Error(mm)    | Accel. Error(mm/s^2) |     MPJVE Error(mm)  |")
    print("+----------------+---------------------+---------------------+----------------------+---------------------+")

    for i, action in enumerate(action_names):
        print(
            "| {0:13} | {1:20} | {2:20} | {3:21} | {4:21} |".format(action, errors_p1[i], errors_p2[i], acceleration_errors[i], errors_mpjve[i]))

    p1 = np.mean(np.array(errors_p1))
    assert round(p1, 4) == round(np.mean(joint_errors),
                                 4), f"MPJPE {p1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(acceleration_errors))
    p2 = np.mean(np.array(errors_p2))
    mpjve = np.mean(np.array(errors_mpjve))

    print("+----------------+---------------------+---------------------+----------------------+---------------------+")
    print("| Average Errors | {0:20} | {1:20} | {2:21} | {3:21} |".format(p1, p2, acceleration_error, mpjve))
    print("+----------------+---------------------+---------------------+----------------------+---------------------+")

    print('per-joint errors:', joint_errors)
    return p1, p2, acceleration_error, mpjve

