import os
import sys
import argparse
from unicodedata import name
import yaml
from datetime import datetime
import wandb
import torch
import numpy as np
import random
from ml_collections import ConfigDict
import time
from absl import flags
from absl import app
import logging
from pathlib import Path

from unot.training import FUNS
from unot.utils.helpers import (
    flat_dict,
    symlink_to_logfile,
    write_metadata,
    fix_random_seeds,
)
from unot.plotting.setup import setup_plt

# FLAGS
flags.DEFINE_boolean("sweep", False, "True if this run a sweep, otherwise False.")
flags.DEFINE_string("outroot", "./results", "Root directory to write model output")
flags.DEFINE_string("outdir", "", "Path to outdir")
flags.DEFINE_string("experiment_name", "", "Name for experiment")
flags.DEFINE_string("experiment_dir", "", "Directory for experiments")
flags.DEFINE_string("config_model", "", "Path to model config file")
flags.DEFINE_string("config_data", "", "Path to data config file")
flags.DEFINE_string("target", None, "Drug target for cell data")
flags.DEFINE_string("model_name", "", "Name of model class")
flags.DEFINE_string("data_class", "", "Class of the dataset")
flags.DEFINE_string("data_name", "", "Name of the dataset")
flags.DEFINE_string("data_name2", "", "Additional name of dataset (eg hours in cell)")
flags.DEFINE_string("run_name", "", "Name of the run (date-time)")
flags.DEFINE_boolean("restart", False, "Delete cache.")


FLAGS = flags.FLAGS


def main(argv):
    _, *unparsed = flags.FLAGS(argv, known_only=True)

    # load configuration
    if not FLAGS.sweep:
        with open(FLAGS.config_model) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        config = ConfigDict(config)

        with open(FLAGS.config_data) as file:
            data_config = yaml.load(file, Loader=yaml.FullLoader)
        config.update(data_config)

        config.config_data = FLAGS.config_data
        config.config_model = FLAGS.config_model
        FLAGS.model_name = config.model.name

        # store names for outdir
        project = config.data.type
        if config.data.type == "cell":
            if FLAGS.target is not None:
                config.data.target = FLAGS.target
            else:
                config.data.target = "trametinib"
            project = project + "-" + config.data.target
            FLAGS.data_class = config.data.type
            FLAGS.data_name = config.data.target
            FLAGS.data_name2 = config.data.path.split("/")[-1].replace(".h5ad", "")

        elif config.data.type == "toy":
            FLAGS.data_class = config.data.type
            FLAGS.data_name = config.data.source + "_" + config.data.target

        wandb_config = flat_dict(config.to_dict())

        wandb.init(
            project=project,
            mode="disabled",
            entity="frederikeluebeck",
            config=wandb_config,
        )

    # name and create output directory
    outdir = name_outdir()
    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    yaml.dump(
        config.to_dict(),
        open(outdir / "config.yaml", "w"),
        default_flow_style=False,
    )

    symlink_to_logfile(outdir / "log")
    write_metadata(outdir / "metadata.json", argv)
    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)

    # if FLAGS.restart, remove existing model checkpoints
    if FLAGS.restart:
        (cachedir / "model.pt").unlink(missing_ok=True)
        (cachedir / "last.pt").unlink(missing_ok=True)

    now = datetime.now()
    dt = now.strftime("%Y-%m-%d-%H-%M-%S-%f")
    wandb.run.name = dt
    FLAGS.run_name = dt

    print("Configuration:", flush=True)
    print(config, flush=True)

    # load training function based on model name
    train = FUNS.get(config.model.name)

    start = time.time()
    train(config, outdir)
    end = time.time()
    print("Execution time: ", (end - start))


def name_outdir():
    expdir = os.path.join(
        FLAGS.outroot,
        FLAGS.data_class,
        FLAGS.experiment_dir,
        FLAGS.data_name,
        FLAGS.data_name2,
        FLAGS.experiment_name,
        f"model-{FLAGS.model_name}",
    )

    return Path(expdir)


if __name__ == "__main__":
    print("Start")

    fix_random_seeds()
    setup_plt()
    main(sys.argv)
