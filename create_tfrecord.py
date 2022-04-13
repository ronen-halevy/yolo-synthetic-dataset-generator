import os
import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class CreateTfrecords:
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

    @staticmethod
    def create_example(image, path, example):
        boxes = np.reshape(example['bboxes'], -1)
        shapes_id = [shape['id'] for shape in example['shapes']]

        feature = {
            "image": CreateTfrecords.image_feature(image),
            "path": CreateTfrecords.bytes_feature(path),
            "xmin": CreateTfrecords.float_feature_list(boxes[0::4].tolist()),
            "ymin": CreateTfrecords.float_feature_list(boxes[1::4].tolist()),
            "xmax": CreateTfrecords.float_feature_list(boxes[2::4].tolist()),
            "ymax": CreateTfrecords.float_feature_list(boxes[3::4].tolist()),
            "category_id": CreateTfrecords.int64_feature_list(shapes_id),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    @staticmethod
    def parse_tfrecord_fn(example):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "path": tf.io.FixedLenFeature([], tf.string),
            "xmin": tf.io.VarLenFeature(tf.float32),
            "ymin": tf.io.VarLenFeature(tf.float32),
            "xmax": tf.io.VarLenFeature(tf.float32),
            "ymax": tf.io.VarLenFeature(tf.float32),
            "category_id": tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        return example


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
    def read_example(self, tfrecords_file_path):
        raw_dataset = tf.data.TFRecordDataset(tfrecords_file_path)
        dataset_example = raw_dataset.map(self.parse_tfrecord_fn)
        return dataset_example




def main():
    # root_dir = "/home/ronen/PycharmProjects/shapes-dataset/"
    tfrecords_out_dir = "dataset/tfrecords_dir"
    # input_images_dir = os.path.join(root_dir, "dataset/annotations/annotations.json")
    input_annotation_file = "dataset/annotations/annotations.json"
    create = CreateTfrecords()
    create.create_tfrecords(input_annotation_file, tfrecords_out_dir)



# def run_example():
    import os
    import glob, os

    sample_file = ''
    for file in os.listdir(tfrecords_out_dir):
        if file.endswith(".tfrec"):
            sample_file = (f"{tfrecords_out_dir}/{file}")
            break

    if sample_file:
        dataset_example = create.read_example(sample_file)

        for features in dataset_example.take(1):
            plt.figure(figsize=(7, 7))
            plt.imshow(features["image"].numpy())

            plt.show()

if __name__ == '__main__':
    main()
