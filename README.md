# Uncertainty Quantification for Graph Neural Networks

02456 - Deep Learning (Fall 2022) @ Technical University of Denmark


### Training models

Advised to run training loop through a terminal via the `run.py`-file. 
For obtaining help on input arguments do:

```
python run.py -h
```

A few examples are provided for training models:

#### Toy dataset (1D) - evidential learning

```
python run.py --mode train --data_dir content/data --dataset TOY1D --batch_size 64 \
              --model TOY1D --epochs 500 --lr 0.05 --loss_function NIG --NIG_lambda 0.001 \
              --val_every_step 1 --tensorboard_logdir logs --experiment_name test \
              --seed 42 --device cuda
```

#### QM7 dataset - evidential learning

```
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 64 \
              --model GNN3D --epochs 500 --lr 0.001 --loss_function NIG --NIG_lambda 0.0 \
              --val_every_step 25 --tensorboard_logdir logs --experiment_name test \
              --seed 42 --device cuda
```


### Evaluating models

*OBS! Currently not supported...*