#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_tfrecord.py
#   Author      : ronen halevy
#   Created date:  4/16/22
#   Description :
#
# ================================================================

import os
import json
import tensorflow as tf
import numpy as np


class dataset_util:
    @staticmethod
    def image_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )

    @staticmethod
    def bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    @staticmethod
    def bytes_feature_list(value):
        """Returns a bytes_list from a string / byte."""
        value = [x.encode('utf8') for x in value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_feature_list(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature_list(value):
        """Returns a list of float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class CreateTfrecords:
    @staticmethod
    def create_example(image, path, example):
        boxes = np.reshape(example['bboxes'], -1)
        id = [object['id'] for object in example['objects']]
        try:
            text = [object['text'] for object in example['objects']]
        except Exception as e:
            pass

        feature = {
            "image/encoded": dataset_util.image_feature(image),
            "image/filename": dataset_util.bytes_feature(path),
            "image/object/bbox/xmin": dataset_util.float_feature_list(boxes[0::4].tolist()),
            "image/object/bbox/ymin": dataset_util.float_feature_list(boxes[1::4].tolist()),
            "image/object/bbox/xmax": dataset_util.float_feature_list(boxes[2::4].tolist()),
            "image/object/bbox/ymax": dataset_util.float_feature_list(boxes[3::4].tolist()),
            "image/object/class/label": dataset_util.int64_feature_list(id),
            'image/object/class/text': dataset_util.bytes_feature_list(text),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def create_tfrecords(self, input_annotation_file, tfrecords_out_dir):
        with open(input_annotation_file, "r") as f:
            annotations = json.load(f)["annotations"]

        print(f"Number of images: {len(annotations)}")

        num_samples = min(4096, len(annotations))
        num_tfrecords = len(annotations) // num_samples
        if len(annotations) % num_samples:
            num_tfrecords += 1  # add one record if there are any remaining samples

        if not os.path.exists(tfrecords_out_dir):
            os.makedirs(tfrecords_out_dir)  # c

        for tfrec_num in range(num_tfrecords):
            samples = annotations[(tfrec_num * num_samples): ((tfrec_num + 1) * num_samples)]

            with tf.io.TFRecordWriter(
                    tfrecords_out_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
            ) as writer:
                for sample in samples:
                    image_path = sample['image_path']  # f"{images_dir}/{sample['image_id']:012d}.jpg"
                    image = tf.io.decode_jpeg(tf.io.read_file(image_path))
                    example = CreateTfrecords.create_example(image, image_path, sample)
                    writer.write(example.SerializeToString())


if __name__ == '__main__':
    tfrecords_out_dir = "dataset/tfrecords"
    input_annotation_file = "dataset/annotations/annotations.json"
    create = CreateTfrecords()
    create.create_tfrecords(input_annotation_file, tfrecords_out_dir)
