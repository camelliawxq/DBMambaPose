# [2025] DBMambaPose: Decoupled Spatial-Temporal Bidirectional State Space Model for Efficient 3D Human Pose Estimation
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This is the official PyTorch implementation of the paper "DBMambaPose: Decoupled Spatial-Temporal Bidirectional State Space Model for Efficient 3D Human Pose Estimation" (2025).

## Environment
The ~~pr~~oject is developed under the following environment:
- Python 3.10.13
- PyTorch 2.1.1
- CUDA 11.8

For installation of the project dependencies, please run:
```
conda create -n DBMambaPose python=3.10.13
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal_conv1d>=1.1.3.post1
pip install mamba-ssm>=1.1.1
pip install -r requirements.txt
```

## Dataset
### Human3.6M
#### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:
```text
python h36m.py  --n-frames 243
```

### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train.py --config <PATH-TO-CONFIG> --new-checkpoint <PATH-TO-CHECKPOINT>
```
where config files are located at `configs/h36m`. 

### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG> --new-checkpoint <PATH-TO-CHECKPOINT>
```
where config files are located at `configs/mpi`. Like Human3.6M, weight and biases can be used.

## Evaluation
| Method         | # frames | # Params | # MACs | H3.6M weights |
|----------------|----------|----------|--------|---------------|
| DBMambaPose-XS | 243      | 1.4M     | 4.2G   |    [download](https://drive.google.com/file/d/1IbHVyMShM2pCNFQ-8xJb8CKAzarrx7Cy/view?usp=sharing)   |
| DBMambaPose-S  | 243      | 4.2M     | 12.4G  |    [download](https://drive.google.com/file/d/1AMxrkWEZo2Rc2B9yhUK9HWk0wHuQkdfj/view?usp=sharing)   |
| DBMambaPose-B  | 243      | 8.7M     | 27.4G  |    [download](https://drive.google.com/file/d/1z_CMaN10e-FIgb-UeqB7q2gqLQUAsHkO/view?usp=sharing)   |
| DBMambaPose-L  | 243      | 12.2M    | 38.3G  |    [download](https://drive.google.com/file/d/1t720ssbsqvs_45sDkJD-KzZelbJ49H3y/view?usp=sharing)   |

After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train.py --eval-only  --config <PATH-TO-CONFIG> --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME>
```
For example if DBMambaPose-L of H.36M is downloaded and put in `checkpoint` directory, then we can run:
```
python train.py --eval-only --config configs/h36m/DBMambaPose-L.yaml --checkpoint checkpoint --checkpoint-file DBMambaPose-L-h36m.pth.tr
```
Similarly, MPI-INF-3DHP can be evaluated as follows:
```
python train_3dhp.py --eval-only --config <PATH-TO-CONFIG> --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME>
```

## Demo
Our demo is a modified version of the one provided by [MHFormer](https://github.com/Vegetebird/MHFormer) repository. First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download our base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory. Then, you need to put your in-the-wild videos in the './demo/video' directory.

Run the command below:
```
python vis_demo.py
```
Sample demo output:

<p align="center"><img src="figure/sample_video.gif" width="60%" alt="" /></p>

## Acknowledgement
Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MotionAGFormer](https://github.com/taatiteam/MotionAGFormer)
- [Mamba](https://github.com/state-spaces/mamba)

We thank the authors for releasing their codes.
