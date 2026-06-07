
# ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing

## :notebook_with_decorative_cover: Code for Paper: ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing [[arXiv]](https://arxiv.org/abs/2507.04678)

<p align="center">
    <img src="./figures/abstract.png" width="95%" height="95%">
</p>

## Abstract <p align="justify">

#### a) ChangeBridge generates post-event remote sensing images from pre-event observations and multimodal spatial controls.
#### b) It models the pre-to-post transition with a spatiotemporal diffusion bridge, improving spatial and temporal coherence.
#### c) It supports controllable scenario generation and can serve as a data generation engine for change detection tasks.

---

## :speech_balloon: ChangeBridge Overview

<p align="center">
    <img src="./figures/method0.png" width="95%" height="95%">
</p>

<p align="center">
    <img src="./figures/method1.png" width="95%" height="95%">
</p>

---

## A. Preparations

### 1. Dataset Structure

```bash
Change detection dataset:
├─A
├─B
├─label
└─list
   ├─train.txt
   ├─val.txt
   └─test.txt
```

`A` denotes pre-event images, `B` denotes post-event images, and `label` denotes change masks.

### 2. Create and activate conda environment

```bash
conda create --name changebridge python=3.8
conda activate changebridge
pip install -r requirements.txt
```

---

## B. Train and Sample

### 1. Train

```bash
python main.py --config configs/your_config.yaml --train
```

For multi-GPU training:

```bash
python main.py --config configs/your_config.yaml --train --gpu_ids 0,1,2,3
```

### 2. Sample

```bash
python main.py --config configs/your_config.yaml --sample
```

Please modify the dataset path, checkpoint path, and sampling settings in the configuration file.

---

## C. Visual Results

<p align="center">
    <img src="./figures/experiment6.png" width="95%" height="95%">
</p>

---

## Citation

If this project is helpful to your research, please kindly cite our paper.

```bibtex
@misc{zhao2025changebridge,
  title         = {ChangeBridge: Spatiotemporal Image Generation with Multimodal Controls for Remote Sensing},
  author        = {Zhenghui Zhao and Chen Wu and Xiangyong Cao and Di Wang and Hongruixuan Chen and Datao Tang and Liangpei Zhang and Zhuo Zheng},
  year          = {2025},
  eprint        = {2507.04678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2507.04678}
}
```
