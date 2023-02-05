import os
import pytorch_lightning as pl
import torch
from typing import Union

from pfam.utils import load_json, save_json, reader, build_vocab, build_labels
from pfam.dataset import preprocess, SequenceDataset
from pfam.model import ProtCNN


class Experiment:
    """
    ML experiment tracking.

    An experiment consists of a particular data processing (which rare amino acids to treat as unknown) and potentially
    several training sub folders. This class handles the tracking of the data processing (by saving fam2label,
    label2fam, word2id and params - e.g. data path, partition, rare_AAs - dictionaries) and the tracking of the
    trainings (by saving tensorboard logs, train_params - e.g. max_len, batch_size, weight_decay and other CLI user
    parameters - and hparams - inputs to LightningModule - parameters). This ensure consistent tracking of flexible ML
    experiments.

    ML experiments are stored as sub folders "experiment_name" inside a folder "experiment_path". If an
    "experiment_name" already exist, its data processing files are loaded, else they are generated and saved.
    A training is tracked and saved in a version_X (X is incremented) sub folder inside "experiment_name".

    Example folder structure:
    └───experiments
    │   └───experimentXYZ
    │   │   |   fam2label.json
    │   │   |   label2fam.json
    │   │   |   params.json
    │   │   |   word2id.json
    │   |   └───version_0
    │   |   |   |   epoch=0-step=1061.ckpt
    │   |   |   |   events.out.tfevents
    │   |   |   |   hparams.yaml
    │   |   |   |   train_params.json
    """

    def __init__(self, experiment_path: str, experiment_name: str, data_path: Union[str, None] = None,
                 partition: Union[str, None] = None, rare_AAs: list = []):
        """

        :param experiment_path: path to folder with experiments
        :param experiment_name: name of experiment, used as sub folder name
        :param data_path: location of the random_split/ data folder
        :param partition: data split to use (train, dev or test)
        :param rare_AAs: list of amino acid letters to treat as unknown <unk>
        """
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        rare_AAs = sorted(list(set(rare_AAs)))

        if not os.path.exists(self.experiment_path + "/" + self.experiment_name):
            os.makedirs(self.experiment_path + "/" + self.experiment_name)
            self.params, self.fam2label, self.label2fam, self.word2id = self.create(data_path, partition, rare_AAs)
            print("Created experiment")
        else:
            self.params, self.fam2label, self.label2fam, self.word2id = self.load()
            assert (data_path is None and partition is None and rare_AAs == []) or \
                   (self.params["data_path"] == data_path and self.params["partition"] == partition and self.params[
                       "rare_AAs"] == rare_AAs)
            print("Loaded experiment")

    def create(self, data_path: str, partition: str, rare_AAs: list):
        """
        Create experiment folder and data processing files (params.json, fam2label.json, word2id.json and label2fam.json)
        :param data_path: location of the random_split/ data folder
        :param partition: data split to use (train, dev or test)
        :param rare_AAs: list of amino acid letters to treat as unknown <unk>
        :return:
        """
        params = {"data_path": data_path, "partition": partition, "rare_AAs": rare_AAs}

        train_data, train_targets = reader(partition, data_path)
        fam2label = build_labels(train_targets)
        label2fam = {v: k for k, v in fam2label.items()}
        word2id = build_vocab(train_data, rare_AAs=set(rare_AAs))
        print(f"There are {len(fam2label)} labels.")
        print(f"AA dictionary formed. The length of dictionary is: {len(word2id)}.")

        save_json(self.experiment_path + "/" + self.experiment_name + "/params.json", params)
        save_json(self.experiment_path + "/" + self.experiment_name + "/fam2label.json", fam2label)
        save_json(self.experiment_path + "/" + self.experiment_name + "/word2id.json", word2id)
        save_json(self.experiment_path + "/" + self.experiment_name + "/label2fam.json", label2fam)

        return params, fam2label, label2fam, word2id

    def load(self):
        """
        Load data processing files (params.json, fam2label.json, word2id.json and label2fam.json) from experiment folder
        :return:
        """
        params = load_json(self.experiment_path + "/" + self.experiment_name + "/params.json")
        fam2label = load_json(self.experiment_path + "/" + self.experiment_name + "/fam2label.json")
        word2id = load_json(self.experiment_path + "/" + self.experiment_name + "/word2id.json")
        label2fam = load_json(self.experiment_path + "/" + self.experiment_name + "/label2fam.json")

        return params, fam2label, label2fam, word2id

    def train(self, seq_max_len: int, batch_size: int, lr: float, momentum: float, weight_decay: float,
              num_workers: int, gpus: int, epochs: int) -> None:
        """

        :param seq_max_len: used to truncate a sequence of amino-acid letters to its first max_len letters or pad to max_len
        :param batch_size: batch size for training
        :param lr: learning rate
        :param momentum: momentum
        :param weight_decay: weight decay
        :param num_workers: number of workers for data loaders
        :param gpus: number of gpu (0 if none)
        :param epochs: number of training epochs
        :return:
        """
        train_params = {
            "seq_max_len": seq_max_len, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
            "momentum": momentum, "num_workers": num_workers, "gpus": gpus, "epochs": epochs
        }

        train_dataset = SequenceDataset(
            word2id=self.word2id, fam2label=self.fam2label, max_len=train_params["seq_max_len"],
            data_path=self.params["data_path"], split="train"
        )
        dev_dataset = SequenceDataset(
            word2id=self.word2id, fam2label=self.fam2label, max_len=train_params["seq_max_len"],
            data_path=self.params["data_path"], split="dev"
        )

        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_params["batch_size"],
            shuffle=True,
            num_workers=train_params["num_workers"],
        )
        dataloaders['dev'] = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=train_params["batch_size"],
            shuffle=False,
            num_workers=train_params["num_workers"],
        )

        prot_cnn = ProtCNN(
            vocab_size=len(self.word2id), num_classes=len(self.fam2label), seq_max_len=seq_max_len,
            lr=train_params["lr"], momentum=train_params["momentum"], weight_decay=train_params["weight_decay"]
        )

        tb_logger = pl.loggers.TensorBoardLogger(save_dir=self.experiment_path, name=self.experiment_name)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.experiment_path + "/" + self.experiment_name + "/version_{}/".format(tb_logger.version),
            monitor="valid_acc",
            mode="max"
        )
        early_stopping = pl.callbacks.EarlyStopping(monitor="valid_acc", mode="max")
        trainer = pl.Trainer(gpus=gpus, max_epochs=epochs, logger=tb_logger,
                             callbacks=[checkpoint_callback, early_stopping])
        trainer.fit(prot_cnn, dataloaders['train'], dataloaders['dev'])

        save_json(self.experiment_path + "/" + self.experiment_name + "/version_{}/train_params.json".format(
            tb_logger.version), train_params)

    def test(self, batch_size: int, num_workers: int, gpus: int, experiment_checkpoint: str) -> None:
        """

        :param batch_size: batch size for testing
        :param num_workers: number of workers for data loaders
        :param gpus: number of gpu (0 if none)
        :param experiment_checkpoint: local path (from experiment sub folder) to model checkpoint (e.g. version_0/epoch=0-step=1061.ckpt)
        :return:
        """
        train_params = load_json(
            self.experiment_path + "/" + self.experiment_name + "/" + experiment_checkpoint.split("/")[
                0] + "/train_params.json")

        test_dataset = SequenceDataset(
            word2id=self.word2id, fam2label=self.fam2label, max_len=train_params["seq_max_len"],
            data_path=self.params["data_path"],
            split="test"
        )

        dataloaders = {}
        dataloaders['test'] = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        prot_cnn = ProtCNN.load_from_checkpoint(
            self.experiment_path + "/" + self.experiment_name + "/" + experiment_checkpoint)
        prot_cnn.eval()

        pl.seed_everything(0)
        trainer = pl.Trainer(gpus=gpus, logger=False)
        trainer.test(prot_cnn, dataloaders['test'])

    def predict(self, sequence: str, experiment_checkpoint: str) -> str:
        """

        :param sequence: sequence of amino acids (a.k.a one protein)
        :param experiment_checkpoint: local path (from experiment sub folder) to model checkpoint (e.g. version_0//epoch=0-step=1061.ckpt)
        :return: protein class
        """
        train_params = load_json(
            self.experiment_path + "/" + self.experiment_name + "/" + experiment_checkpoint.split("/")[
                0] + "/train_params.json")

        prot_cnn = ProtCNN.load_from_checkpoint(
            self.experiment_path + "/" + self.experiment_name + "/" + experiment_checkpoint)
        prot_cnn.eval()

        x = preprocess(sequence, train_params["seq_max_len"], self.word2id).unsqueeze(0)
        y = prot_cnn(x).argmax(axis=1)

        return self.label2fam[str(y.numpy()[0])]