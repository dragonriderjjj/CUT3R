### Installation

1. Clone CUT3R.

```bash
git clone https://github.com/dragonriderjjj/CUT3R.git
cd CUT3R
```

2. Create the environment.

```bash
conda create -n cut3r python=3.11 cmake=3.14.0
conda activate cut3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt

# for training logging, you can skip it for eval
pip install git+https://github.com/nerfstudio-project/gsplat.git
# for evaluation
pip install evo
pip install open3d
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).

```bash
cd src/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```

### Download Checkpoints

We use ckpt saved on Google Drive:

To download the weights, run the following commands:

```bash
cd src

# for 512 dpt ckpt
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
cd ..
```
### Download dataset
for 7scenes evaluation:
place the data under "CUT3R/data/"
```bash
mkdir data
cd data

# download 7scenes data 
gdown [path-to-download-link]
unzip ./7scenes.zip
cd ..
```

### Multi-view Reconstruction

```bash
bash eval/mv_recon/run.sh # You may need to change [--num_processes] to the number of your gpus
```

Results will be saved in `eval_results/mv_recon/${model_name}_${ckpt_name}/
