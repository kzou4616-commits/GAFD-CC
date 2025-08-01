# GAFD-CC: Global-Aware Feature Decoupling with Confidence Calibration for OOD Detection

## Getting Start

### 1. Install dependencies

read requirements.txt

### 2. Download datasets

- Large-scale OOD datasets: 
    - In-distribution dataset: **ImageNet-1k**
    - Out-of-distribution datasets: **iNaturalist, SUN, Places, Textures, OpenImages-O, ImageNet-O, SSB-hard, NINCO**
- Small-scale OOD datasets:
    - In-distribution dataset: **CIFAR-10, CIFAR-100**
    - Out-of-distribution datasets: **SVHN, LSUN_crop, LSUN_resize, iSUN, Places, Textures**

- Download the datasets and put them in the `data` folder.

cifar100 Test OOD Datasets
* [SVHN, LSUN, iSUN, Textures, Places365, ImageNet_resize](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

imagenet1k Test OOD Datasets:
refer to [KNN](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

### 3. Baseline  methods

- ViM and other methods implementation we refer to [CADRef](https://github.com/LingAndZero/CADRef/tree/main/ood_methods)
- 

### 4. Run the code

```bash for running GAFD-CC
python ood_ours.py --OOD_method gafdcc --gpu 0 --bs 32  --ind_dataset ImageNet --model vit --ood_dataset iNat SUN Places Textures openimage_o imagenet_o
```
```bash for running baseline  methods, replace the ood_method 
python ood_eval.py --OOD_method ood_method --gpu 0 --bs 32  --ind_dataset ImageNet --model vit --ood_dataset iNat SUN Places Textures openimage_o imagenet_o
```

options:
  -h, --help            show this help message and exit
  --ind_dataset IND_DATASET
                        in-distribution dataset name
  --ood_dataset OOD_DATASET [OOD_DATASET ...]
                        OOD dataset list
  --model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}
                        model name
  --gpu GPU             gpu id
  --num_classes NUM_CLASSES
                        number of classes
  --random_seed RANDOM_SEED
                        random seed
  --bs BS               batch size
  --OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef}
                        OOD method name
  --use_feature_cache USE_FEATURE_CACHE
                        use feature cache
  --use_score_cache USE_SCORE_CACHE
                        use score cache
  --cache_dir CACHE_DIR
                        cache directory
  --result_dir RESULT_DIR
                        result directory
  --num_workers NUM_WORKERS
                        number of workers
  --logit_method {Energy,MSP,MaxLogit,GEN}
                        logit method for CADRef
```# GAFD-CC: Global-Aware Feature Decoupling with Confidence Calibration for OOD Detection

## Getting Start

### 1. Install dependencies

```bash

```

### 2. Download datasets

- Large-scale OOD datasets: 
    - In-distribution dataset: **ImageNet-1k**
    - Out-of-distribution datasets: **iNaturalist, SUN, Places, Textures, OpenImages-O, ImageNet-O, SSB-hard, NINCO**
- Small-scale OOD datasets:
    - In-distribution dataset: **CIFAR-10, CIFAR-100**
    - Out-of-distribution datasets: **SVHN, LSUN_crop, LSUN_resize, iSUN, Places, Textures**

- Download the datasets and put them in the `data` folder.

cifar100 Test OOD Datasets
* [SVHN, LSUN, iSUN, Textures, Places365, ImageNet_resize](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

imagenet1k Test OOD Datasets:
refer to [KNN](https://github.com/deeplearning-wisc/knn-ood?tab=readme-ov-file)

### 3. Baseline  methods

- ViM and other methods implementation we refer to [CADRef](https://github.com/LingAndZero/CADRef/tree/main/ood_methods)
- 

### 4. Run the code

```bash for running GAFD-CC
python ood_ours.py --OOD_method gafdcc --gpu 0 --bs 32  --ind_dataset ImageNet --model vit --ood_dataset iNat SUN Places Textures openimage_o imagenet_o
```
```bash for running baseline  methods, replace the ood_method 
python ood_eval.py --OOD_method ood_method --gpu 0 --bs 32  --ind_dataset ImageNet --model vit --ood_dataset iNat SUN Places Textures openimage_o imagenet_o
```

options:
  -h, --help            show this help message and exit
  --ind_dataset IND_DATASET
                        in-distribution dataset name
  --ood_dataset OOD_DATASET [OOD_DATASET ...]
                        OOD dataset list
  --model {resnet,vit,convnext,densenet,regnet,efficientnet,swin}
                        model name
  --gpu GPU             gpu id
  --num_classes NUM_CLASSES
                        number of classes
  --random_seed RANDOM_SEED
                        random seed
  --bs BS               batch size
  --OOD_method {MSP,ODIN,Energy,GEN,ReAct,DICE,GradNorm,MaxLogit,ASH,OptFS,VIM,Residual,CARef,CADRef}
                        OOD method name
  --use_feature_cache USE_FEATURE_CACHE
                        use feature cache
  --use_score_cache USE_SCORE_CACHE
                        use score cache
  --cache_dir CACHE_DIR
                        cache directory
  --result_dir RESULT_DIR
                        result directory
  --num_workers NUM_WORKERS
                        number of workers
  --logit_method {Energy,MSP,MaxLogit,GEN}
                        logit method for CADRef
```
