# Edge Weight Prediction For Category-Agnostic Pose Estimation
<a href="https://orhir.github.io/edge_cape/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href=""><img src="https://img.shields.io/badge/arXiv-311.17891-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt">
<img src="https://img.shields.io/badge/License-Apache-yellow"></a>


By [Or Hirschorn](https://scholar.google.co.il/citations?user=GgFuT_QAAAAJ&hl=iw&oi=ao) and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)

This repo is the official implementation of "[Edge Weight Prediction For Category-Agnostic Pose Estimation
]()".

# Hugging Face Demo Coming Soon!
### Stay tuned for the upcoming demo release!


## 🔔 News
- **`25 November 2024`** Initial Code Release


## Introduction
Given only one example image and skeleton,  our method refines the skeleton to enhance pose estimation on unseen categories.

Using our method, given a support image and skeleton we can refine the structure for better pose estimation on images from unseen categories.

## Citation
Please consider citing our paper and GraphCape if you found our work useful:
```bibtex
@misc{hirschorn2024edgeweightpredictioncategoryagnostic,
      title={Edge Weight Prediction For Category-Agnostic Pose Estimation}, 
      author={Or Hirschorn and Shai Avidan},
      year={2024},
      eprint={2411.16665},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.16665}, 
}

@misc{hirschorn2023pose,
      title={A Graph-Based Approach for Category-Agnostic Pose Estimation}, 
      author={Or Hirschorn and Shai Avidan},
      year={2024},
      eprint={2311.17891},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.17891}, 
}
```

## Getting Started

### Docker [Recommended]
We provide a docker image for easy use.
You can simply pull the docker image from docker hub, containing all the required libraries and packages:

```
docker pull orhir/edgecape
docker run --name edgecape -v {DATA_DIR}:/workspace/EdgeCape/EdgeCape/data/mp100 -it orhir/edgecape /bin/bash
```
### Conda Environment
We train and evaluate our model on Python 3.8 and Pytorch 2.0.1 with CUDA 12.1. 

Please first install pytorch and torchvision following official documentation Pytorch. 
Then, follow [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html) to install the following packages:
```
mmcv-full=1.7.2
mmpose=0.29.0
```
Having installed these packages, run:
```
python setup.py develop
```

## MP-100 Dataset
Please follow the [official guide](https://github.com/orhir/PoseAnything) to prepare the MP-100 dataset for training and evaluation, and organize the data structure properly.

## Training

### Training
To train the model, run:
```
python run.py --config [path_to_config_file]  --work_dir [path_to_work_dir]
```

## Evaluation and Pretrained Models

### Evaluation
The evaluation on a single GPU will take approximately 30 min. 

To evaluate the pretrained model, run:
```
python test.py [path_to_config_file] [path_to_pretrained_ckpt]
```

### Pretrained Models

You can download the pretrained models from following [link](https://drive.google.com/drive/folders/1gbeeVQ-Y8Dj2FrsDatf5ZLWpzv5u8HyL?usp=sharing).

## Acknowledgement

Our code is based on code from:
 - [MMPose](https://github.com/open-mmlab/mmpose)
 - [PoseAnything](https://github.com/orhir/PoseAnything)


## License
This project is released under the Apache 2.0 license.