#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : read_shapes_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#
# ================================================================

import os
import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class ReadTfrecords:
    @staticmethod
    def prepare_sample(features):
        x_train = tf.image.resize(features['image'], size=(416, 416))
        labels = tf.cast(features['category_id'], tf.float32)
        y_train = tf.stack([features['xmin'], features['ymin'], features['xmax'], features['ymax'], labels], axis=1)
        return x_train, y_train

    @staticmethod
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "xmin": tf.io.VarLenFeature(tf.float32),
            "ymin": tf.io.VarLenFeature(tf.float32),
            "xmax": tf.io.VarLenFeature(tf.float32),
            "ymax": tf.io.VarLenFeature(tf.float32),
            "text": tf.io.VarLenFeature(tf.string),
            "category_id": tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
        example['xmin'] = tf.sparse.to_dense(example['xmin'])
        example['ymin'] = tf.sparse.to_dense(example['ymin'])
        example['xmax'] = tf.sparse.to_dense(example['xmax'])
        example['ymax'] = tf.sparse.to_dense(example['ymax'])
        example['category_id'] = tf.sparse.to_dense(example['category_id'])
        example['text'] = tf.sparse.to_dense(example['text'])

        return example

    @staticmethod
    def get_dataset(filenames, batch_size):
        dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
            .map(ReadTfrecords.parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .map(ReadTfrecords.prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(batch_size * 10)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return dataset


if __name__ == '__main__':
    tfrecords_out_dir = 'dataset/tfrecords'
    train_filenames = tf.io.gfile.glob(f'{tfrecords_out_dir}/*.tfrec')
    batch_size = 32
    read = ReadTfrecords()
    ds = read.get_dataset(train_filenames, batch_size)
