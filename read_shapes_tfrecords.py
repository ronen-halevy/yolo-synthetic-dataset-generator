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
        labels = tf.cast(features['label'], tf.float32)
        y_train = tf.stack([features['xmin'], features['ymin'], features['xmax'], features['ymax'], labels], axis=1)
        return x_train, y_train

    @staticmethod
    def parse_tfrecord_fn(example):
        feature_description = {
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/filename": tf.io.FixedLenFeature([], tf.string),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['image'] = tf.io.decode_jpeg(example['image/encoded'], channels=3)
        example['xmin'] = tf.sparse.to_dense(example['image/object/bbox/xmin'])
        example['ymin'] = tf.sparse.to_dense(example['image/object/bbox/ymin'])
        example['xmax'] = tf.sparse.to_dense(example['image/object/bbox/xmax'])
        example['ymax'] = tf.sparse.to_dense(example['image/object/bbox/ymax'])
        example['label'] = tf.sparse.to_dense(example['image/object/class/label'])
        example['text'] = tf.sparse.to_dense(example['image/object/class/text'])

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
