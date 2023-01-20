# Uncertainty Quantification for Graph Neural Networks 

**02456 - Deep Learning (Fall 2022) @ Technical University of Denmark**

This project examines uncertainty quantification in graph neural networks by approaching the concept of 
*Evidential Learning*. The main components of this repository are:

- Code structure for experimenting with training models (see *Training models* section below)
- [Notebook containing descriptions for reproducibility of the associated project report.](https://nbviewer.org/github/albertkjoller/uq-gnn/blob/main/explainer_notebook.ipynb)

For in-depth considerations and conclusions of results consult the associated paper.

### Setup 

Run the following:
```
pip install -r requirements.txt
```
Additionally installing Pytorch is required - check their webpage for finding a compatible version (GPU or CPU).

### Data

You can find the data [here](https://drive.google.com/drive/folders/1cSR59Bb4Tj_FiLei4866AD1-4GMr7dop?usp=sharing).
Download the `data`-folder and place it in the `content`-directory!

## Training models

Advised to run training loop through a terminal via the `run.py`-file. 
For obtaining help on input arguments do:

```
python run.py -h
```

A few examples are provided for training models:

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
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 64 \
              --model EvidentialQM7_3D --epochs 200 --lr 5e-3 --loss_function NIG \
	      --NIG_lambda 0.75 --kappa 1.0 --kappa_decay 0.99 --scalar 'none' \
	      --val_every_step 25 --tensorboard_logdir logs --experiment_name EVIDENTIAL_QM7 \
	      --seed 0 --device cuda
```

#### Gaussian MLE (baseline)

```
python run.py --mode train --data_dir content/data --dataset QM7 --batch_size 64 \
              --model testbase --epochs 200 --lr 5e-3 --loss_function GAUSSIANNLL \
	      --kappa 0.0 --kappa_decay 1.0 --scalar 'standardize' \
	      --val_every_step 25 --tensorboard_logdir logs --experiment_name BASELINE_QM7 \
              --save_path models --seed 0 --device cuda
```


#### Example of training a 1D model - Evidential Learning

Epistemic              |  Aleatoric              |  Parameters
:-------------------------:|:-------------------------:|:-------------------------:|
![epistemic_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/epistemic.gif)  |  ![aleatoric_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/aleatoric.gif)  |  ![parameters_uq](https://github.com/albertkjoller/uq-gnn/blob/main/figures/parameters.gif)



### Evaluating models

When running in evaluation mode, the resulting figures will be saved in a folder called `eval_{experiment_name}`. 
Please look here after running the commands.

#### Toy Dataset - 1D (example)

We can run the trained 1D model (with name REPRODUCTION_TOY) in evaluation mode for obtaining plots and tabular results
by running the following command:

```
python run.py --mode evaluation --data_dir content/data --batch_size 64 --save_path models \
	        --NIG_lambda 0.01 --scalar 'none' --seed 0 --device cpu \
	        --experiment_name REPRODUCTION_TOY --model TOY1D --dataset TOY1D --id_ood ID	      
```

If we want to compare performance on ID with OOD test set the command would be the following:
```
python run.py --mode evaluation --data_dir content/data --batch_size 64 --save_path models \
	        --NIG_lambda 0.01 --seed 0 --device cpu \
	        --experiment_name REPRODUCTION_TOY --model TOY1D --dataset TOY1D --id_ood ID --scalar 'none'  \
	        --experiment_name REPRODUCTION_TOY --model TOY1D --dataset TOY1D-OOD --id_ood OOD --scalar 'none' 
```


#### General evaluation commands:

It is possible to evaluate models across e.g. seed to see variance of performance, type of models to compare them, and also different data types like ID or OOD to evaluate the epistemic uncertainty. The general running command line is:

```
python run.py --mode evaluation --data_dir content/data --batch_size 64 --save_path models --seed 0 --device cpu
```
You can evaluate an experiment by appending the following to the command above along with the relevant input arguments related to the experiment:
(*OBS: Remember to change the argument values!*)

```
--experiment_name model_1 --model modelType --dataset dataset_1 --id_ood ?
```

And in order to add another experiment or dataset in the evaluation, simply replicate the line above and adjust the necessary parameters. Below is an example of appending the same model but trained on a different seed,
(*OBS: Experiment_name has changed*):

```
--experiment_name model_2 --model modelType --dataset dataset_2 --id_ood ?
```


### Overview of input arguments

- `--mode [train, evaluation]`, defines whether to run in train or evaluation mode.

if train:
	
- `--epochs int`, How many epochs to run.
- `--batch_size int`, Batch size to be used.
- `--lr float`, Learning rate when training.
- `--val_every_step int`, Frequency of running model on validation data during training.
- `--save_path str`, Path to saving models.
- `--model str`, The model type to use when training. Currently, either 'TOY1D', 'GNN3D', 'BASE1D', or 'BASE3D'.
- `--loss_function [NIG, MSE, GAUSSIANNLL]`, Type of loss function.
	- if NIG:
	- `--NIG_lambda float`, Lambda value when running NIG loss function, if multiple values are stated it enables a train-loop
- `--kappa float`, Trade-off between specific loss and RMSE. Must be in range [0, 1]. A value of 1 means full emphasis on the RMSE.
- `--kappa_decay float`, Decay parameter for threshold value between loss and RMSE. Must be in range [0, 1] with 1 being no decay.
- `--scalar ['standardize', 'none']`, Type of standardization on target variable.

if evaluation:
- `--id_ood ['ID', 'OOD']`, For evaluation only: whether dataset is in or out of distribution.
	
for both train and evaluation:
- `--seed int`, Seed for pseudo-random behaviour.
- `--device ['cpu', 'cuda']`, Device to run computations on.
- `--data_dir str`, Path to data directory where various datasets are stored.
- `--tensorboard_logdir str`, Path to location for storing tensorboard log-files.
- `--experiment_name str`, Name of experiment run (should identical for train and evaluation mode).
- `--dataset [TOY1D, QM7, SYNTHETIC]`, Name of the dataset to be used. 
	- if TOY1D (and training):
	- `--toy_noise_level float`, Noise level when generating the data.
	- `--N_points int`, Number of points for TOY1D dataset.

