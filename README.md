This is a PyTorch implementation of our paper.  We present weight-based ensemble method for crop pest identification: VecEnsemble and MatEnsenble. Our method achieved the highest accuracy of 77.39% on the large-scale complex-scene IP102 dataset.



# 1. Requirements

    python=3.8
    torch==1.11.0
    torchvision==0.12.0
    timm==0.6.11
    scikit-learn



data prepare: IP102 dataset with the following folder structure.

    │IP102/
    ├──train/
    │  ├── Adristyrannus
    │  │   ├── 64003.jpg
    │  │   ├── 64006.jpg
    │  │   ├── ......
    │  ├── ......
    ├──val/
    │  ├── Adristyrannus
    │  │   ├── 64011.jpg
    │  │   ├── 64012.jpg
    │  │   ├── ......
    │  ├── ......


# 2. Train basic models

Take fine-tuning ResNet-50, ViT-S/16, Volo-d1 and ViP-Small/7 on the IP102 dataset as an example (the pre-trained ResNet-50 and ViT-S/16 come from the timm library, and the pre-trained Volo-d1 and ViP-Small/7 come from their authors, with download links [volo_d1](https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar) and [vip_s7](https://drive.google.com/file/d/1cX6eauDrsGsLSZnqsX7cl0oiKX8Dzv5z/view?usp=sharing), respectively.) :


<details>
<summary>
  train ResNet-50
</summary>

  	python3 train_basic_model.py /path/to/IP102 --num-classes 102 --img-size 224 --model resnet50 --pretrained --epochs 200 --opt adamw --lr 2e-3  --min-lr 1e-5 --sched cosine -b 128 --drop-path 0.2 --warmup-epochs 20 --weight-decay 5e-4

</details>

<details>
<summary>
  train ViT-S/16
</summary>

  	python3 train_basic_model.py /path/to/IP102 --num-classes 102 --img-size 224 --model vit_small_patch16_224 --pretrained --epochs 200 --opt adamw --lr 1e-4 --min-lr 1e-5 --sched cosine -b 64  --drop-path 0.2 --weight-decay 5e-4

</details>

<details>
<summary>
  train Volo-d1
</summary>

  	python3 train_basic_model.py /path/to/IP102 --num-classes 102 --model volo_d1 --img-size 224 -b 64 --lr 8.0e-6 --min-lr 4.0e-6 --drop-path 0.1 --epochs 200 --apex-amp --weight-decay 1.0e-8 --warmup-epochs 5 --finetune /path/to/pre-trained-volo_d1

</details>

<details>
<summary>
  train ViP-Small/7
</summary>

  	python3 train_basic_model.py /path/to/IP102 --num-classes 102 --model vip_s7 -b 64 --opt adamw --epochs 200 --sched cosine --apex-amp --img-size 224 --drop-path 0.1 --lr 2e-3 --min-lr 1e-5 --weight-decay 0.05  --warmup-epochs 20 --finetune /path/to/pre-trained-vip_s7

</details>




# 3. Validation

To evaluate our basic models, run:

<details>
<summary>
  evaluate ResNet-50
</summary>

  	python3 validate.py /path/to/IP102 --split test --model resnet50 --num-classes 102 --img-size 224 --checkpoint /path/to/checkpoint --no-test-pool -b 128

</details>

<details>
<summary>
  evaluate ViT-S/16
</summary>

  	python3 validate.py /path/to/IP102 --split test --model vit_small_patch16_224 --num-classes 102 --img-size 224 --checkpoint /path/to/checkpoint --no-test-pool -b 64

</details>

<details>
<summary>
  evaluate Volo-d1
</summary>

  	python3 validate.py /path/to/IP102 --split test --model volo_d1 --num-classes 102 --img-size 224 --checkpoint /path/to/checkpoint --no-test-pool -b 64

</details>


<details>
<summary>
  evaluate ViP-Small/7
</summary>

  	python3 validate.py /path/to/IP102 --split test --model vip_s7 --num-classes 102 --img-size 224 --checkpoint /path/to/checkpoint --no-test-pool -b 64

</details>



| Model | Accuracy | Download |
|:-----:|:-------:|:-----:|
| Resnet-50 | 74.57 | [link](https://drive.google.com/file/d/1SHycW-ITMP69NcY2OmwdYeOuPITjUC2-/view?usp=drive_link) |
| ViT-S/16 | 75.13 | [link](https://drive.google.com/file/d/14YFeB2LZpDYa2fDyc5t3ddVKw6QhlUgR/view?usp=drive_link) |
| Volo-d1 | 76.20 | [link](https://drive.google.com/file/d/1vhiCG7mAY7hwq82eZtH_jVz1Jia5tbpH/view?usp=drive_link) |
| ViP-Small/7 | 73.41 | [link](https://drive.google.com/file/d/1V86SWj8CVFcF3JjYPn8FoFC7mYlf2vdJ/view?usp=drive_link) |

All basic models are trained and evaluated on a single TITAN Xp 12G GPU.



# 4. Ensemble methods

Write the names and paths of the basic models to be integrated into `. /tools/model_list.py`, and run `record_outputs.py` to get the outputs of all basic models on the validation and test sets:

	python3 record_outputs.py /path/to/IP102 --num-classes 102

Calculate vector-based weights for VecEnsemble and evaluate the recognition performance of VecEnsemble:

	python3 VecEnsemble.py

Calculate matrix-based weights for MatEnsemble and evaluate the recognition performance of MatEnsemble:

	python3 MatEnsemble.py

Ablation experiments with four ensemble methods (Hard Voting, Soft Voting, VecEnsemble and MatEnsemble):

```
python3 ablation_exp.py
```

