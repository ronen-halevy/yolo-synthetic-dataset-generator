#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_all.py
#   Author      : ronen halevy 
#   Created date:  8/23/22
#   Description :
#
# ================================================================
import yaml
import argparse

from src.create_shapes_dataset import create_dataset
from src.create_class_names_file import create_class_mames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.yaml',
                        help='yaml config file')

    parser.add_argument("--shapes_file", type=str, default='config/shapes.yaml',
                        help='shapes yaml config file')

    parser.add_argument("--class_names_out_file", type=str, default='dataset/class.names',
                        help='class_names output _file')

    args = parser.parse_args()
    config_file = args.config_file
    shapes_file = args.shapes_file
    class_names_out_file = args.class_names_out_file

    try:
        create_dataset(config_file, shapes_file)
        create_class_mames(config_file=config_file, shapes_in_file=shapes_file,
                           class_names_out_file=class_names_out_file)
    except Exception as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
