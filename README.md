# ViT Edition for [Latent Intrinsics Emerge from Training to Relight (NeurIPS 2024, Spotlight)](https://arxiv.org/abs/2405.21074)

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/JesusmiCaH/Latent_Intrinsics#
cd Latent_Intrinsics

# 2. (Recommended) Create and activate environment
conda create -n latent_intrinsics python=3.11 -y
conda activate latent_intrinsics

# 3. Install dependencies (adjust torch/cu version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 5. Download multi-illumination dataset
mkdir -p data
cd data
wget https://data.csail.mit.edu/multilum/multi_illumination_train_mip2_jpg.zip 
unzip multi_illumination_train_mip2_jpg.zip && rm multi_illumination_train_mip2_jpg.zip
cd ..

# 6. Download pretrained checkpoint
wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

# 7. Run Baseline 
data_path=data
port=50000
python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} main_cls.py \
--data_path ${data_path} \
--reg_weight 1e-4 \
--intrinsics_loss_weight 1e-1 \
--epochs 120 \
--batch_size 16 \
--learning_rate 2e-4 \
--weight_decay 1e-2 \
--resume

# 8. Run ViT-Ver Latent Intrinsic Encoder
data_path=data
port=50000
python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=${port} main_cls_ViT.py \
--data_path ${data_path} \
--reg_weight 1e-4 \
--intrinsics_loss_weight 1e-1 \
--epochs 120 \
--batch_size 16 \
--learning_rate 2e-4 \
--weight_decay 1e-2 \
--resume
```
### Notes

- To accelerate the training, I have early stopped both the Dataset loading and training per epoch. You may change them to normal in `utils.py/parallel_load_image` and `main_cls_ViT.py/train_D`
## Training

To train the model, download the [Multi-illumination dataset](https://projects.csail.mit.edu/illumination/) and update the `data_path` in `srun.sh`. Then, use `bash srun.sh` to launch the training script for 4 GPUs.

## Evaluation

### 1. Pretrained Checkpoint

We provide the pre-trained model to infer the albedo.  
Download the pretrained relighting model from this [Google Drive Link](https://drive.google.com/file/d/1bb4Up7SNZ9lBTku4LGAJe49wE4bEVlBo/view?usp=sharing).

### 2. Albedo Evaluation

- Download the [IIW dataset](http://opensurfaces.cs.cornell.edu/publications/intrinsic/).
- Update the `data_path` and `checkpoint_path` in `srun_albedo.sh`.
- Infer the albedo and calculate WHDR by sweeping the threshold with `bash srun_albedo.sh`.

### 3. Relight Evaluation

- Update the `data_path` and `checkpoint_path` in `srun_relight.sh`.
- Run `bash srun_albedo.sh` to evaluate the relighting images with arbitrary references.
