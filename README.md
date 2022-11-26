# Uncertainty Quantification for Graph Neural Networks

02456 - Deep Learning (Fall 2022) @ Technical University of Denmark

### Setup 

Run the following:
```
pip install -r requirements.txt
```
Additionally installing Pytorch is required - check their webpage for finding a compatible version (GPU or CPU).

## Training models

Advised to run training loop through a terminal via the `run.py`-file. 
For obtaining help on input arguments do:

```
python run.py -h
```

A few examples are provided for training models:

### Toy Dataset - 1D
#### Evidential learning

![epistemic_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/epistemic.gif)

***OBS***: *For reproducing [Amini et al.](https://arxiv.org/pdf/1910.02600.pdf), run with these
arguments:*

```
python run.py --mode train --data_dir content/data --dataset TOY1D --batch_size 128 \
              --model TOY1D --epochs 5000 --lr 5e-3 --loss_function NIG --NIG_lambda 0.01 \
              --val_every_step 50 --tensorboard_logdir logs --experiment_name REPRODUCTION \
              --seed 0 --device cuda
```


#### Simple Baseline (Work-In-Progress...)

```
python run.py --mode train --data_dir content/data --dataset TOY1D --batch_size 128 \
              --model BASE1D --epochs 5000 --lr 0.1 --loss_function RMSE \
              --val_every_step 5 --tensorboard_logdir logs --experiment_name BASELINE \
              --seed 0 --devie cuda
```

### Molecular Graph Datasets - 3D
#### QM7 dataset - evidential learning

```
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 64 \
              --model GNN3D --epochs 500 --lr 0.001 --loss_function NIG --NIG_lambda 0.0 \
              --val_every_step 25 --tensorboard_logdir logs --experiment_name test \
              --seed 42 --device cuda
```


### Evaluating models

*OBS! Currently not supported...*