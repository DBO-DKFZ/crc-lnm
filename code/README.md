<p align="center">
  <strong>hip</strong> - <i>A deep learning pipeline for histology using Pytorch-Lightning.</i>
</p>

----------------

### Installation

1. Create conda environment: 
````
conda create -n hip python=3.8
conda activate hip
````
2. Install hip + requirements:
````
Install PyTorch > 1.8 manually first
cd hip
pip install -r requirements.txt -e .
````

This will install hip in editable mode, meaning any of your changes to the source are available next time you import hip.

3. Download pre-trained model [here](https://ngc.nvidia.com/catalog/models/nvidia:med:clara_pt_pathology_metastasis_detection) and put it as *nvidia-resnet18.pt* in models/.


### Train
````
python hip/scripts/tile_train.py --cfg path/to/config.cfg --gpus <id>
````

This will create *logs* in your current working directory e.g.
````
logs
└── resnext_100
    ├── ckpts
    │   └── 2021-01-15-15-03-38-resnext50_32x4d-epoch=00-valid_auroc=0.93valid_loss_epoch_tile=0.57-valid_loss_epoch_slide=0.49.ckpt
    ├── version_0
    │   ├── events.out.tfevents.1610719242.CAD660710.52555.0
    │   └── hparams.yaml
    └── version_1
        ├── events.out.tfevents.1610719420.CAD660710.52919.0
        └── hparams.yaml
````

### Test
*Tile-Level*
````
python hip/scripts/tile_test.py ---cfg path/to/config.cfg --ckpt path/to/checkpoint.ckpt --gpus <id>
````
