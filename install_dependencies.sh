#source $(conda info --root)/etc/profile.d/conda.sh

conda create -y --name nubot python=3.8.5
conda activate nubot

conda install -y -c conda-forge jaxlib
conda install -y -c conda-forge jax

pip install -r requirements.txt

pip install -e .