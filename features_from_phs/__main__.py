from .gen_features import gen_features
from .cluster_labels import cluster_labels
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(input_file)
parser.add_argument(output_file)


def main():
    args = parser.parse_args()

    print_progress(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
