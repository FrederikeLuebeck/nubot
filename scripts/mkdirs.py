from genericpath import exists
import os
from pathlib import Path
import os
import yaml

from ml_collections import ConfigDict

if __name__ == "__main__":

    TIME = "24:00"
    DEBUG = False

    CONFIG_MODELS = [
        "./config/models/ubot.yaml",
        "./config/models/ot.yaml",
        # "./config/models/gaussian_approx.yaml"
    ]
    CONFIG_DATAS = [
        "./config/data/4i-8h-raw.yaml",
        "./config/data/4i-24h-raw.yaml",
    ]
    EXPERIMENT_DIR = "ubot"  # "0922_submission"
    EXPERIMENT_NAME = "ubot"
    OUTROOT = "./results/"  # "/cluster/scratch/fluebeck/results/rebuttal"
    DRUGS = [
        "cisplatin",
        "crizotinib",
        "dabrafenib",
        "dacarbazine",
        "dasatinib",
        "decitabine",
        "dexamethasone",
        "erlotinib",
        "everolimus",
        "hydroxyurea",
        "imatinib",
        "ixazomib",
        "lenalidomide",
        "melphalan",
        "midostaurin",
        "olaparib",
        "paclitaxel",
        "palbociclib",
        "panobinostat",
        "regorafenib",
        "sorafenib",
        "staurosporine",
        "temozolomide",
        "trametinib",
        "ulixertinib",
        "vindesine",
    ]
    for drug in DRUGS:
        for CONFIG_DATA in CONFIG_DATAS:
            for CONFIG_MODEL in CONFIG_MODELS:
                name = "model-" + CONFIG_MODEL.split("/")[-1].replace(".yaml", "")
                timestep = CONFIG_DATA.split("/")[-1].replace("4i-", "")
                timestep = timestep.replace("-raw.yaml", "") + "_subm"

                new_dir = (
                    Path(OUTROOT)
                    / EXPERIMENT_DIR
                    / drug
                    / timestep
                    / EXPERIMENT_NAME
                    / name
                )
                print(new_dir)
                new_dir.mkdir(exist_ok=True, parents=True)
                with open(CONFIG_MODEL) as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)

                with open(CONFIG_MODEL) as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)
                config = ConfigDict(config)

                with open(CONFIG_DATA) as file:
                    data_config = yaml.load(file, Loader=yaml.FullLoader)
                config.update(data_config)

                config.data.target = drug

                yaml.dump(
                    config.to_dict(),
                    open(new_dir / "config.yaml", "w"),
                    default_flow_style=False,
                )
