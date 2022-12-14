from collections import MutableMapping
import yaml
from ml_collections import ConfigDict
from pathlib import Path
import os
import re
import json
from time import localtime, strftime
import torch
import random
import numpy as np


def load_config(path, unparsed=None):

    path = Path(path)

    if path.exists():
        config = ConfigDict(yaml.load(open(path, "r"), yaml.SafeLoader))
    else:
        config = ConfigDict()

    if unparsed is not None:
        opts = parse_cli_opts(unparsed)
        config.update(opts)

    return config


def dump_config(path, config):
    yaml.dump(
        config.to_dict(), open(path, "w"), default_flow_style=False, Dumper=yaml.Dumper
    )
    return


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def split_rec(k, v, out, sep=".", as_dot_dict=False):
    # splitting keys in dict, calling recursively to break items on '.'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, DotDict() if as_dot_dict else dict()))
    else:
        out[k] = v


def nest_dict(d, sep=".", as_dot_dict=False):
    result = DotDict() if as_dot_dict else dict()
    for k, v in d.items():
        # for each key split_rec splits keys to form recursively nested dict
        split_rec(k, v, result, sep=sep, as_dot_dict=as_dot_dict)
    return result


def flat_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def symlink_to_logfile(loglink):
    loglink = Path(loglink)

    lsb_outputfile = os.getenv("LSB_OUTPUTFILE")
    jobid = os.getenv("LSB_JOBID")
    if lsb_outputfile is None or jobid is None:
        return

    logpath = Path(lsb_outputfile).resolve()
    if lsb_outputfile.endswith("/") or logpath.is_dir():
        logpath = logpath / f"{jobid}.out"

    if loglink.exists():
        loglink.unlink()
    try:
        loglink.symlink_to(logpath)
    except FileExistsError:
        pass

    with open(loglink.with_name(".loghistory"), "a") as fp:
        fp.write(f"{logpath}\n")

    return


def parse_cli_opts(args):
    def parse(argiter):
        for opt in argiter:
            if "=" not in opt:
                value = next(argiter)
                key = re.match(r"^--config\.(?P<key>.*)", opt)["key"]
                # ^start, --config, \. special character ., (?P<key>.*) symbolic group makes whatever is found in () accessible via key

            else:
                match = re.match(r"^--config\.(?P<key>.*)=(?P<value>.*)", opt)
                key, value = match["key"], match["value"]

            value = yaml.load(value, Loader=yaml.UnsafeLoader)
            yield key, value

    opts = dict()
    if len(args) == 0:
        return opts

    argiter = iter(args)
    for key, val in parse(argiter):
        *tree, leaf = key.split(".")
        lut = opts
        for k in tree:
            dict_ref = lut.setdefault(
                k, dict()
            )  # set default returns a reference to the new dict
            lut = dict_ref
        lut[leaf] = val

    return opts


def write_metadata(outpath, argv):
    program, *args = argv
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    lut = {
        "startedAt": now,
        "args": args,
        "program": program,
    }

    json.dump(lut, open(outpath, "w"), indent=2)
    return


def parse_config_cli(path, args):
    if isinstance(path, list):
        config = ConfigDict()
        for arg in path:
            config.update(yaml.load(open(arg), Loader=yaml.RoundTripLoader))
    else:
        config = load_config(path)

    opts = parse_cli_opts(args)
    config.update(opts)

    return config


def load_item_from_save(path, key, default):
    path = Path(path)
    if not path.exists():
        return default

    ckpt = torch.load(path)
    if key not in ckpt:
        return default

    return ckpt[key]


def fix_random_seeds(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
