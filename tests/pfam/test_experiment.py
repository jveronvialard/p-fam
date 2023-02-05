from pfam.experiment import Experiment


class TestDataset:

    def test_create_load(self, test_data_path):
        _ = Experiment(
            experiment_name="XYZ",
            experiment_path=test_data_path + "/experiments",
            data_path=test_data_path + "/random_split",
            partition="train",
            rare_AAs=[]
        )
        _ = Experiment(
            experiment_name="XYZ",
            experiment_path=test_data_path + "/experiments",
            data_path=test_data_path + "/random_split",
            partition="train",
            rare_AAs=[]
        )
