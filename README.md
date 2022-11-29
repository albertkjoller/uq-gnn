# Uncertainty Quantification for Graph Neural Networks

**02456 - Deep Learning (Fall 2022) @ Technical University of Denmark**

This project examines uncertainty quantification in graph neural networks by approaching the concept of 
*Evidential Learning*. The main components of this repository are:

- [Notebook containing descriptions for reproducibility of the associated project report.](https://nbviewer.org/github/albertkjoller/uq-gnn/blob/main/explainer.ipynb#2)
- Code structure for experimenting with training models (see *Training models* section below)


#### Example - Evidential Learning

Epistemic              |  Aleatoric              |  Parameters
:-------------------------:|:-------------------------:|:-------------------------:|
![epistemic_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/epistemic.gif)  |  ![aleatoric_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/aleatoric.gif)  |  ![parameters_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/parameters.gif)


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

### Data

You can find the data [here](https://drive.google.com/drive/folders/1cSR59Bb4Tj_FiLei4866AD1-4GMr7dop?usp=sharing).
Download the `data`-folder and place it in the `content`-directory!

### Toy Dataset - 1D
#### Evidential learning

***OBS***: *For reproducing [Amini et al.](https://arxiv.org/pdf/1910.02600.pdf), run with the 
arguments below. We use 625 epochs which equals 5000 iterations as they argue, as --> iterations / (observations / observations pr. iteration), i.e. 5000 / (1024 / 128) = 625 epochs.*

```
python run.py --mode train --data_dir content/data --dataset TOY1D --batch_size 128 --N_points 1024 \
              --model TOY1D --epochs 625 --lr 5e-3 --loss_function NIG --NIG_lambda 0.01 \
              --val_every_step 10 --tensorboard_logdir logs --experiment_name REPRODUCTION_TOY \
              --save_path models --seed 0 --device cuda
```


#### Gaussian MLE (baseline)

```
python run.py --mode train --data_dir content/data --dataset TOY1D --batch_size 128 --N_points 1024 \
              --model BASE1D --epochs 625 --lr 5e-3 --loss_function GAUSSIANNLL \
              --val_every_step 10 --tensorboard_logdir logs --experiment_name BASELINE_TOY \
              --save_path models --seed 0 --device cuda
```

### Molecular Graph Datasets - 3D
#### Evidential learning - QM7 Dataset

```
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 128 \
              --model GNN3D --epochs 1000 --lr 5e-3 --loss_function NIG --NIG_lambda 0.01 \
              --val_every_step 25 --tensorboard_logdir logs --experiment_name EVIDENTIAL_QM7 \
              --seed 0 --device cuda
```

#### Gaussian MLE (baseline)

```
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 128 \
              --model BASE3D --epochs 1000 --lr 5e-3 --loss_function GAUSSIANNLL \
              --val_every_step 25 --tensorboard_logdir logs --experiment_name BASELINE_QM7 \
              --save_path models --seed 0 --device cuda
```


### Evaluating models


*OBS! Currently not fully supported...*

```
python run.py --mode evaluation --data_dir content/data --dataset TOY1D --batch_size 128 \
              --model TOY1D --experiment_name REPRODUCTION \
              --save_path models --seed 0
```
