"""
Visualize mixture model results in bulk.
"""

import argparse
import pathlib

from visualize_mixture_results import main as visualize

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
       'result_dir',
       help='directory containing result subdirectories from fitting mixture models.'
       )
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    directory = pathlib.Path(args.result_dir)

    for r in directory.iterdir():
        visualize(r)
