# [ArXiv] Fine-gained Air Quality Inference of Low-data-quality Sensor Data using Self-supervised Learning

This is a PyTorch implementation of Multi-task Spatio-temporal Network (**MTSTN**), as described in our paper: **[Fine-gained Air Quality Inference of Low-data-quality Sensor Data using Self-supervised Learning]**, ArXiv.

## Requirements

Our code is based on Python version 3.8.18 and PyTorch version 2.1.0. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Train & Test

You can train and test **MTSTN** through the following commands for 3 datasets.

```shell
python main.py --dataset no2  --batchsize 38 --timestep 26 --heads [13, 4, 1] --dropout [0.75, 0.15, 0.75] --layers [3, 1, 1] --embedding dim [43, 88, 18, 8, 1, 16] --lr 0.001 --weight decay 6e-7 --reduced dimension 1 --loss weight 6e3 --regularization weight [0.3, 0.2] 
python main.py --dataset o3  --batchsize 38 --timestep 34 --heads [13, 4, 1] --dropout [0.75, 0.15, 0.75] --layers [3, 1, 1] --embedding dim [43, 88, 18, 8, 1, 16] --lr 0.0004 --weight decay 6e-7 --reduced dimension 1 --loss weight 6e3 --regularization weight [0.2, 0.3] 
python main.py --dataset pm2_5  --batchsize 38 --timestep 36 --heads [13, 6, 1] --dropout [0.75, 0.15, 0.75] --layers [3, 1, 1] --embedding dim [44, 88, 18, 8, 1, 16] --lr 0.001 --weight decay 2e-8 --reduced dimension 1 --loss weight 5e2 --regularization weight [0.2, 0.3] 
```

## Cite

If you find the paper useful, please cite as following:

```
```

