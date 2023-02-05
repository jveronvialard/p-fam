# p-fam
Protein classifier.
## Requirements
Create a data/ folder at the project root directory.
Download [PFAM dataset](https://www.kaggle.com/datasets/googleai/pfam-seed-random-split) and extract random_split/ 
folder inside the data/ folder.
```
project
│   ...
└───data
│   └───random_split
│   |   └───dev
│   |   |   |   ...
│   |   └───test
│   |   |   |   ...
│   |   └───train
│   |   |   |   ...
```
## ML Experiment Tracking
An experiment consists of 
- a particular data processing (which rare amino acids to treat as unknown): by saving fam2label, label2fam, word2id and
 params - e.g. data path, partition, rare_AAs - dictionaries
- tracking of each training: by saving tensorboard logs, train_params - e.g. max_len, batch_size, weight_decay and other 
CLI user parameters - and hparams - inputs to LightningModule - parameters

to ensure consistent tracking and flexibility.

ML experiments are stored as sub folders "experiment_name" inside a folder "experiment_path". If an
"experiment_name" already exist, its data processing files are loaded, else they are generated and saved.
Each training is tracked and saved in a version_X (X is incremented) sub folder inside "experiment_name".
## Local
Environment setup
```shell script
conda create -n pfamvenv python=3.7.4
conda activate pfamvenv
conda install pytorch==1.8.1 -c pytorch
pip install -e .
```
Model training (only experiment_path, experiment_name and data_dir are required)
```shell script
python train.py --experiment_path ./experiments/ --experiment_name BOUXZ --data_dir ./data/random_split --partition train --rare_AAs B,O,U,X,Z --seq_max_len 120 --batch_size 1024 --lr 1e-2 --momentum 0.9 --weight_decay 1e-2 --num_workers 0 --gpus 1 --epochs 1
```
After having trained a model: model testing (only experiment_path, experiment_name and experiment_checkpoint are 
required), model prediction (all arguments required), and tensorboard visualization
```shell script
python test.py --experiment_path ./experiments/ --experiment_name BOUXZ --experiment_checkpoint version_0/epoch=14-step=15929.ckpt --batch_size 1024 --num_workers 0 --gpus 1
python predict.py --sequence LLQKKIRVRPNRAQLVQRHILDDT --experiment_path ./experiments/ --experiment_name BOUXZ --experiment_checkpoint version_0/epoch=14-step=15929.ckpt
tensorboard --logdir ./experiments/BOUXZ/version_0
```
Unit testing
```shell script
pytest tests
```
## Project Structure
```
project
│   .gitignore
│   Dockerfile
│   predict.py
│   README.md
│   requirements.txt
│   setup.py
│   test.py
│   train.py
└───data
│   └───random_split
│   |   └───dev
│   |   |   |   ...
│   |   └───test
│   |   |   |   ...
│   |   └───train
│   |   |   |   ...
└───experiments
│   └───BOUXZ
│   │   |   fam2label.json
│   │   |   label2fam.json
│   │   |   params.json
│   │   |   word2id.json
│   |   └───version_0
│   |   |   |   epoch=14-step=15929.ckpt
│   |   |   |   events.out.tfevents
│   |   |   |   hparams.yaml
│   |   |   |   train_params.json
└───pfam
│   │   dataset.py
│   │   experiment.py
│   │   model.py
│   │   utils.py
└───tests
│   │   __init__.py
│   │   conftest.py
│   └───data
│   |   └───experiments
│   |   |   └───BOUXZ
│   |   └───random_split
│   |   |   └───train
│   └───pfam
│       │   __init__.py
│       │   test_dataset.py
│       │   test_experiment.py
│       │   test_model.py
│       │   test_utils.py
```