import json
import argparse


def main():
    args = setup_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.

    if args["config"] is not None:
        param = load_json(args["config"])
        args.update(param)  # Add parameters from json

    from trainer import train
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional json file of settings. Command-line defaults are used when omitted.')

    # Global experiment settings.
    parser.add_argument('--prefix', type=str, default='reproduce')
    parser.add_argument('--dataset', type=str, default='ave')
    parser.add_argument('--memory_size', type=int, default=340)
    parser.add_argument('--memory_per_class', type=int, default=20)
    parser.add_argument('--fixed_memory', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--init_cls', type=int, default=7)
    parser.add_argument('--increment', type=int, default=7)
    parser.add_argument('--model_name', type=str, default='avcil', choices=['avcil', 'my', 'moe'])
    parser.add_argument('--convnet_type', type=str, default='resnet32')
    parser.add_argument('--device', nargs='+', default=['0'])
    parser.add_argument('--seed', nargs='+', type=int, default=[42])
    parser.add_argument('--project', type=str, default='nips26')
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--csv_name', type=str, default='default')

    # Distributed training settings.
    parser.add_argument('--distributed', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--local_rank', type=int, default=0)

    # Parameters used by av_cil.py and mymodel.py.
    parser.add_argument('--init_epoch', type=int, default=200)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--init_milestones', nargs='+', type=int, default=[60, 120, 170])
    parser.add_argument('--init_lr_decay', type=float, default=0.1)
    parser.add_argument('--init_weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--milestones', nargs='+', type=int, default=[100])
    parser.add_argument('--lrate_decay', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--T', type=float, default=2)
    parser.add_argument('--instance_contrastive_temperature', type=float, default=0.05)
    parser.add_argument('--class_contrastive_temperature', type=float, default=0.05)

    # Defaults for network variants that may be selected through convnet_type.
    parser.add_argument('--proj_hidden_dim', type=int, default=768)
    parser.add_argument('--proj_output_dim', type=int, default=768)
    parser.add_argument('--init_interpolation_factor', type=float, default=0.5)
    parser.add_argument('--attn_num_heads', type=int, default=8)

    return parser


if __name__ == '__main__':
    main()
