# Introduction
There are many datasets available arround, but I needed a simple and felxible-modifiable dataset for my image classification and object detection experiments.
This is not just a dataset of randomly selected shape images, but a flexible tool which permit an easy modification of the produced images,  done by editing two json files, as detailed here below.

Meanwhile, heres an example image:

![alt text](https://github.com/ronen-halevy/shapes-dataset/blob/main/docs/009898.jpg)


#Dataset Structure
This repo contains both the dataset generation code and a generated dataset which consists of Train, Test, and Validation sections, each holds images jpegs and bounding box data.

#Configuration Files
The dataset contents is configured using 2 json files:

- shapes.json: Defines the set of supported shapes
- config.json: Defines the arrangement of the dataset images and of the shapes within the images.


**shapes.json:**

```json
{
  "image_size": [
    413,
    413
  ],
  "max_objects_in_image": 11,
  "x_diameter_choices": [
    30,
    60,
    100
  ],
  "bg_color": [
    0,
    0,
    0
  ],
  "annotations_font_size": 10,
  "annotatons_text_color": [
    255,
    255,
    255
  ],
  "margin_from_edge": 5,
  "iou_thresh": 0,
  "sections": {
    "train": {
      "num_of_examples": 10000,
      "images_dir": "./shapes-dataset/train/images",
      "annotations_path": "./shapes-dataset/train/annotations/annotations.txt"
    },
    "test": {
      "num_of_examples": 2000,
      "images_dir": "./shapes-dataset/test/images",
      "annotations_path": "./shapes-dataset/train/annotations/annotations.txt"
    },
    "valid": {
      "num_of_examples": 2000,
      "images_dir": "./shapes-dataset/valid/images",
      "annotations_path": "./shapes-dataset/valid/annotations/annotations.txt"
    }
  }
}
```







<!-- ![image](https://user-images.githubusercontent.com/13983042/156055596-289b6692-523e-4496-b807-143db62654a3.png)
 -->





, each in a randomly selected size.

So here is the shapes dataset



A lot of wonderful datasets are now available online, such as COCO or Imagenet. which is challenging the limits of computer vision. But it's not easy for us to do some small experiments with such a large number of images to quickly test the validity of algorithmn. For this reason, I created a small dataset named "yymnist" to do both classification and object detection.
