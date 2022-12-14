import os
from pathlib import Path
import os

if __name__ == "__main__":

    TIME = "24:00"
    DEBUG = False

    CONFIG_MODELS = [
        "./config/models/cell_experiment/nubot.yaml",
        "./config/models/cell_experiment/cellot.yaml",
        "./config/models/cell_experiment/ubot_gan.yaml",
    ]
    CONFIG_DATAS = [
        "./config/data/4i-8h-raw.yaml",
        # "./config/data/4i-24h-raw.yaml",
    ]

    EXPERIMENT_DIR = "1312_reproduced"
    OUTROOT = "/cluster/scratch/fluebeck/results/reproduced"
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
    eval = True
    if not eval:
        for drug in DRUGS:
            for CONFIG_MODEL in CONFIG_MODELS:
                for CONFIG_DATA in CONFIG_DATAS:

                    EXPERIMENT_NAME = CONFIG_MODEL.split("/")[-1].replace(".yaml", "")

                    command = ""
                    if DEBUG:
                        command += "echo "  # test output

                    command += "bsub"

                    # time
                    command += " -W " + TIME
                    # memory per cpu and select one gpu
                    command += ' -R "rusage[mem=40000]"'

                    command += (
                        " 'python scripts/run_model.py --config_model "
                        + CONFIG_MODEL
                        + " --config_data "
                        + CONFIG_DATA
                        + " --target "
                        + drug
                        + " --experiment_name "
                        + EXPERIMENT_NAME
                        + " --experiment_dir "
                        + EXPERIMENT_DIR
                        + " --restart "
                        + " --outroot "
                        + OUTROOT
                        + "'"
                    )

                    print(command)
                    os.system(command)

    if eval:
        outroot = Path("/cluster/scratch/fluebeck/results/reproduced/cell")

        for exp_dir in ["1312_reproduced"]:
            for drug in DRUGS:
                outdir = outroot / exp_dir / drug

                if outdir.is_dir():
                    for data in os.listdir(outdir):
                        if data in ["8h_subm", "24h_subm"]:
                            e = outdir / data
                            if e.is_dir():
                                for exp_name in os.listdir(outdir / data):
                                    if exp_name in ["nubot", "cellot", "ubot_gan"]:
                                        d = outdir / data / exp_name
                                        for model_name in os.listdir(d):
                                            if model_name.startswith("model-"):
                                                model_dir = str(d / model_name)
                                                command = ""
                                                command += "bsub"
                                                command += " -W " + TIME
                                                command += ' -R "rusage[mem=40000]"'
                                                command += (
                                                    " 'python scripts/run_evaluation.py"
                                                    " --outdir "
                                                    + model_dir
                                                    # + " --plot_UMAPs"
                                                    # + " --save_csv "
                                                    + "'"
                                                )
                                                print(command)
                                                os.system(command)
