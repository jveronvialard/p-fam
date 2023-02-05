import argparse

from pfam.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training command line parameters")
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--partition", type=str, default="train")
    parser.add_argument('--rare_AAs', '--list', type=str, default="B,O,U,X,Z")
    parser.add_argument("--seq_max_len", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()
    print(args)

    experiment = Experiment(
        experiment_name=args.experiment_name,
        experiment_path=args.experiment_path,
        data_path=args.data_dir,
        partition=args.partition,
        rare_AAs=args.rare_AAs.split(',')
    )
    experiment.train(
        seq_max_len=args.seq_max_len, batch_size=args.batch_size, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, num_workers=args.num_workers, gpus=args.gpus, epochs=args.epochs
    )
