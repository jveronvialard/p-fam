import argparse

from pfam.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing command line parameters")
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--experiment_checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    args = parser.parse_args()
    print(args)

    experiment = Experiment(
        experiment_name=args.experiment_name, experiment_path=args.experiment_path
    )
    experiment.test(
        batch_size=args.batch_size, num_workers=args.num_workers, gpus=args.gpus,
        experiment_checkpoint=args.experiment_checkpoint
    )
