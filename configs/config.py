import json
from dotmap import DotMap
import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file',
    )
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    return config
