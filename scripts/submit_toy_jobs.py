import os
import subprocess
import sys

# bsub -W 24:00 -R "rusage[mem=40000]" "python scripts/run_model.py --target vindesine"

from pathlib import Path
import os

if __name__ == "__main__":
    eval = True
    TIME = "24:00"
    DEBUG = False

    # CONFIG_MODELS = [
    #     # "./config/models/nubot_v1.yaml",
    #     # "./config/models/gan.yaml",
    #     # "./config/models/cellot2.yaml",
    #     # # "./config/models/cellot.yaml",
    #     "./config/models/rebuttal_toy/nubot_base.yaml",
    #     "./config/models/rebuttal_toy/nubot_reg_0p01.yaml",
    #     "./config/models/rebuttal_toy/nubot_reg_m_0p1.yaml",
    #     # "./config/models/rebuttal_toy/nubot_reg_m_0p5.yaml",
    #     # "./config/models/rebuttal_toy/nubot_reg_m_1.yaml",
    #     # "./config/models/rebuttal_toy/nubot_reg_m_0p5.yaml",
    # ]
    CONFIG_DATAS = [
        # "./config/data/toy-triangle-balanced.yaml",
        "./config/data/toy-triangle-unbalanced-1.yaml",
        "./config/data/toy-triangle-unbalanced-3.yaml",
        # "./config/data/toy-triangle-both-sides-unbalanced.yaml",
    ]
    # EXPERIMENT_NAME = ""
    EXPERIMENT_DIR = "rebuttal_toy_bs400"
    OUTROOT = "/cluster/scratch/fluebeck/results"
    if not eval:
        # for CONFIG_MODEL in CONFIG_MODELS:
        for c in os.listdir("./config/models/rebuttal_toy"):
            if c.startswith("nubot"):
                CONFIG_MODEL = str(Path("./config/models/rebuttal_toy") / c)
                for CONFIG_DATA in CONFIG_DATAS:
                    command = ""
                    if DEBUG:
                        command += "echo "  # test output

                    # CONFIG_MODEL.split("/")[-1].replace(".yaml", "")
                    command += "bsub"

                    # time
                    command += " -W " + TIME
                    # memory per cpu and select one gpu
                    command += ' -R "rusage[mem=40000]"'

                    experiment_name = CONFIG_MODEL.split("/")[-1].replace(".yaml", "")

                    command += (
                        " 'python scripts/run_model.py --config_model "
                        + CONFIG_MODEL
                        + " --config_data "
                        + CONFIG_DATA
                        + " --experiment_dir "
                        + EXPERIMENT_DIR
                        + " --experiment_name "
                        + experiment_name
                        + " --restart "
                        + " --outroot "
                        + OUTROOT
                        + "'"
                    )

                    print(command)
                    os.system(command)
    if eval:
        exp_dirs = [
            "rebuttal_toy_bs400",
            "rebuttal_toy_bs350",
            "rebuttal_toy_bs300",
            "rebuttal_toy_bs250",
        ]
        for EXPERIMENT_DIR in exp_dirs:
            for setup in os.listdir(Path(OUTROOT) / EXPERIMENT_DIR):
                s = Path(OUTROOT) / EXPERIMENT_DIR / setup
                if s.is_dir():
                    for co in os.listdir(Path(s)):
                        if True:  # co  "nubot_reg_m_0p1":
                            p = Path(s) / co
                            for model in os.listdir(p):
                                if model.startswith("model-"):
                                    od = str(Path(p) / model)
                                    command = ""
                                    command += "bsub"
                                    # time
                                    command += " -W " + TIME
                                    # memory per cpu and select one gpu
                                    command += ' -R "rusage[mem=40000]"'

                                    command += (
                                        " 'python scripts/run_toy_evaluation.py"
                                        " --config_model  --outdir "
                                        + od
                                        + "'"
                                    )
                                    print(command)
                                    os.system(command)
