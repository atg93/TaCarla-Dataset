# Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs. 

### Finetune Pretrained Model on BDD10k

To finetune pretrained model on BDD10k, some parameters in config.yaml file should be changed as follows:

```
aspect_ratio_group_factor: -1
data_path: /datasets
dataset: bdd10k
coco_pretrained: [checkpoint path]
exclude_pretrained_params: ['mask_head.kernel_head.cls_branch*']
```
Parameters matched with regular expressions in exclude_pretrained_params will not be loaded to the model.
Default of the parameter is `['(?!backbone)']` (body+fpn will be loaded only). 
```
python -m torch.distributed.launch -nproc_per_node=4 --use_env train.py
 --cfg ../../settings/SOLOv2/solov2-regnet_1_6gf-fpn-base.yaml --output-dir [where you want]
```


