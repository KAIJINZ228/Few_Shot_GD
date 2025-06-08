# 🥇 NTIRE 2025 CD-FSOD Challenge @ CVPR Workshop

We are the **award-winning team** of the **NTIRE 2025 Cross-Domain Few-Shot Object Detection (CD-FSOD) Challenge** at the **CVPR Workshop**.

- 🏆 **Track**: `open-source track`
- 🎖️ **Award**: **1st Place**

🔗 [NTIRE 2025 Official Website](https://cvlai.net/ntire/2025/)  
🔗 [NTIRE 2025 Challenge Website](https://codalab.lisn.upsaclay.fr/competitions/21851)  
🔗 [CD-FSOD Challenge Repository](https://github.com/lovelyqian/NTIRE2025_CDFSOD)

![CD-FSOD Task](https://upload-images.jianshu.io/upload_images/9933353-3d7be0d924bd4270.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---

## 🧠 Overview

This repository contains our solution for the `open-source track` of the NTIRE 2025 CD-FSOD Challenge.  
We propose a method that integrates the mixture-of-experts(MoE) into Grounding DINO, which achieves strong performance on the challenge. 

---

## 🛠️ Environment Setup

```
python 3.8.10
torch 2.0.1
transformers 4.33.1
MultiScaleDeformableAttention 1.0
numpy 1.22.2
```


## 📂 Dataset Preparation
Please follow the instructions in the [official CD-FSOD repo](https://github.com/lovelyqian/NTIRE2025_CDFSOD) to download and prepare the dataset.

## 🏋️ Training
To train the model: 
```
python3 main_train.py
```
Stage2 training is still on its way, coming soon. 

pretrained model: The pre-trained checkpoint can be downloaded from the official website of Grounding DINO(https://github.com/IDEA-Research/GroundingDINO)

## 🔍 Inference & Evaluation
Run inference:
```
python3 inference_a_img.py
```

## 📄 Citation
If you use our method or codes in your research, please cite:
```
@inproceedings{fu2025ntire, 
  title={NTIRE 2025 challenge on cross-domain few-shot object detection: methods and results,
  author={Fu, Yuqian and Qiu, Xingyu and Ren, Bin and Fu, Yanwei and Timofte, Radu and Sebe, Nicu and Yang, Ming-Hsuan and Van Gool, Luc and others},
  booktitle={CVPRW},
  year={2025}
}
```






