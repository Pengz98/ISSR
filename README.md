## 1. Dependency Setup
Our codes are tested on both RTX3090 and RTX4090 GPUs.

Install external dependencies by running:

```pip install torch==2.0.1+cu117 numpy==1.26.4 matplotlib==3.9.0 scikit-learn==1.5.0 seaborn==0.13.2 open3d=0.18.0```

Make sure that all of the dependencies including the Open3D-ML are successfully installed.

Put the official Open3D-ML inside the folder, so as to utilize the configuration it provides.

Download the pre-trained weights following the guidance in Open3D-ML/model_zoo.md, and put those weights in Open3D-ML/logs.


## 2. Dataset Preparation
Follow the instructions in [Open3D-ML](https://github.com/isl-org/Open3D-ML) to download the datasets and pre-trained weights. Check the 'dataset_path' in the main python file (ISSR.py).

## 3. Run Interactive Semantic Segmentation Refinement
Run the following commends to start interactive semantic segmentation on S3DIS, ScanNet and SemanticKITTI respectively.
Change the interact_mode from 'real' to 'simulated' to automatically evaluate the performance with our interaction simulator.

S3DIS:

```python ISSR.py --exp s3dis_pointtransformer --filter --warm_up --entropy_threshold_increase 0.03 --entropy_threshold_decrease 0.03 --beta 100 --lr 1e-3 --interact_mode real```

ScanNet:

```python ISSR.py --exp scannet_sparseconvunet --filter --warm_up --entropy_threshold_increase 0.1 --entropy_threshold_decrease 0.01 --beta 100 --lr 1e-3 --interact_mode real```

SemanticKITTI:

```python ISSR.py --exp semantickitti_randlanet --filter --warm_up --entropy_threshold_increase 0.1 --entropy_threshold_decrease 0.1 --beta 1000 --lr 1e-2 --interact_mode real```


## 4. User Guidance:

- The input scene would appear in the first round.
The ground truth and error map would appear in each interactive round to help users select the optimal point to click, though they are unnecessary in practical use.
The current prediction mask would be appear, and wait for further corrective clicks to facilitate refinement.

- Users could put clicks on the error regions of current mask based on the error map, where the correct labels will automatically assigned to the clicked points based on the ground truth.

- When the current mask has achieved expectance, users could turn to the next scene through double clicking on any point in the current mask.

## Acknowledgement
This repository builds upon the excellent work of [Open3D-ML](https://github.com/isl-org/Open3D-ML), which provides abundant off-the-shelf semantic segmentation networks pre-trained on indoor and outdoor point cloud datasets.
