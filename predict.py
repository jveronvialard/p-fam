import argparse

from pfam.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict command line parameters")
    parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--experiment_checkpoint", type=str, required=True)
    args = parser.parse_args()

    args = parser.parse_args()
    print(args)

    experiment = Experiment(
        experiment_name=args.experiment_name,
        experiment_path=args.experiment_path
    )
    predicted_class = experiment.predict(
        sequence=args.sequence, experiment_checkpoint=args.experiment_checkpoint
    )
    print(predicted_class)
