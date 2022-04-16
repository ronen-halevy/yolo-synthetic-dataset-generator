#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : run_all.py
#   Author      : ronen halevy 
#   Created date:  4/16/22
#   Description :
#
# ================================================================

import numpy as np
from PIL import Image, ImageDraw

import tensorflow as tf
import json
import create_shapes_dataset
import create_tfrecord
import read_shapes_tfrecords

if __name__ == '__main__':
    config_file = 'config.json'
    shapes_file = 'shapes.json'
    with open(config_file) as f:
        config_data = json.load(f)

    with open(shapes_file) as f:
        shapes_data = json.load(f)['shapes']

    create_shapes_dataset.create_dataset(config=config_data, shapes=shapes_data)
##

    tfrecords_out_dir = "dataset/tfrecords"
    # input_images_dir = os.path.join(root_dir, "dataset/annotations/annotations.json")
    input_annotation_file = "dataset/annotations/annotations.json"

    create = create_tfrecord.CreateTfrecordsShapes()
    create.create_tfrecords(input_annotation_file, tfrecords_out_dir)

    tfrecords_out_dir = "dataset/tfrecords"
    train_filenames = tf.io.gfile.glob(f"{tfrecords_out_dir}/*.tfrec")
    batch_size = 10  # 32

    read = read_shapes_tfrecords.ReadTfrecordsShapes()
    ds = read.get_dataset(train_filenames, batch_size)
