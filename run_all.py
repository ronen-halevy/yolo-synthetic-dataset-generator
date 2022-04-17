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


import tensorflow as tf
import json
import create_shapes_dataset
import create_tfrecord
import read_shapes_tfrecords

if __name__ == '__main__':
    # 1. Create Shapes Dataset. Output: image jpeg files and annotations.json with bbox metadata

    config_file = 'config.json'
    shapes_file = 'shapes.json'
    with open(config_file) as f:
        config_data = json.load(f)

    with open(shapes_file) as f:
        shapes_data = json.load(f)['shapes']

    create_shapes_dataset.create_dataset(config=config_data, shapes=shapes_data)

    # 2. Create tfrecords. Each entry is an image amd metadata. Number of bounding boxes varies between images
    tfrecords_out_dir = "dataset/tfrecords"
    input_annotation_file = "dataset/annotations/annotations.json"

    create = create_tfrecord.CreateTfrecords()
    create.create_tfrecords(input_annotation_file, tfrecords_out_dir)

    # 3. Demonstrate dataset read

    tfrecords_out_dir = "dataset/tfrecords"
    train_filenames = tf.io.gfile.glob(f"{tfrecords_out_dir}/*.tfrec")
    batch_size = 10  # 32

    read = read_shapes_tfrecords.ReadTfrecords()
    ds = read.get_dataset(train_filenames, batch_size)
