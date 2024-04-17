import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go TPMS")
    parser.add_argument('--rev-info-path', type=str, help="path to reviewers information file")
    parser.add_argument('--pap-info-path', type=str, help="path to paper information file")
    parser.add_argument('--train-inter-path', type=str, help="path to interaction files between reviewers and papers they have reviewed")
    parser.add_argument('--test-inter-path', type=str, help="path to interaction files between reviewers and papers they will be reviewing")
    return parser.parse_args()