# Introduction
There are many datasets available arround, but I needed a simple and felxible-modifiable dataset for my image classification and object detection experiments.
This is not just a dataset of randomly selected shape images, but a flexible tool which permit an easy modification of the produced images,  done by editing two json files, as detailed here below.

Meanwhile, heres an example image:

![alt text](https://github.com/ronen-halevy/shapes-dataset/blob/main/docs/009898.jpg)


#Dataset Structure
This repo contains both the dataset generation code and a generated dataset which consists of Train, Test, and Validation sections, each holds images jpegs and bounding box data.

#Configuration Files
The dataset contents is configured using 2 json files:

- config.json: Defines the arrangement of the dataset images and of the shapes within the images.
- shapes.json: Defines the set of supported shapes

Here's the config.hson:

**config.json:**

Some orientation about the file's attributes:

-`image_size`: Image size is set uniformly for all images. 
-`shape_width_choices`: Shapes' width is randomly selected from this list.
- `bg_color`: Images background color. The shapes are laid on top of this background.
- `annotations_font_size`: Font size of image annotation text.
- `annotatons_text_color`: Color of annotation text.
- `margin_from_edge`: Minimal distance in pixels of from a shape's edge to the image's edge.
- `iou_thresh`: Maximal iou permitted between any shape. Settin iou to 0 would mean no overlap between shapes.
- `sections`: Sections of dataset, each assigned with related attributes as detailed next. Current implementation defines 3 sections:`train`, `test` and `valid`, (i.e. validation).
- `num_of_examples`: Size of section's dataset
- `images_dir`: Directory for placing images.
- `annotations_path`: Path to annotation file. This is a per section file which lists dataset's images with their corresponding shapes' classes and binding boxes. 

```json
{
  "image_size": [
    413,
    413
  ],
  "max_objects_in_image": 11,
  "shape_width_choices": [
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

**shapes.json**

This file is a list of shapes, each defined by a set of attributes:
`id`: This is the class id used by the classification network. Different shapes can be bound together, e.g. setting same id to all shapes would result in a single common class.
`shape_type`: A generalized definition of the shape, needed by the draw pocedure.
`name`: This is the class name used by the display annotations.
`shape_aspect_ratio`: Shape's height to width ratio.
`fill_color`: Shape's fill color. Leave empty to produce a frame without a fill.
`outline_color`: Shape's frame color.

This list can be expended or decreased, by just adding or removing a shape record.
As noted before, shapes can be grouped together to a common id, No need for any more configuration.

```json
{
  "shapes": [
    {
      "id": 0,
      "shape_type": "ellipse",
      "name": "circle",
      "shape_aspect_ratio": 1,
      "fill_color": [
        255,
        0,
        255
      ],
      "outline_color": [
        255,
        0,
        255
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 1,
      "shape_type": "ellipse",
      "name": "wide_ellipse",
      "shape_aspect_ratio": 0.5,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 2,
      "shape_type": "ellipse",
      "name": "narrow_ellipse",
      "shape_aspect_ratio": 2,
      "fill_color": [
        190,
        80,
        20
      ],
      "outline_color": [
        190,
        80,
        20
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 3,
      "shape_type": "rectangle",
      "name": "square",
      "shape_aspect_ratio": 1,
      "fill_color": [
        220,
        0,
        90
      ],
      "outline_color": [
        220,
        0,
        90
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 4,
      "shape_type": "rectangle",
      "name": "wide_rectangle",
      "shape_aspect_ratio": 0.5,
      "fill_color": [
        12,
        88,
        20
      ],
      "outline_color": [
        12,
        88,
        20
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 5,
      "shape_type": "rectangle",
      "name": "narrow_rectangle",
      "shape_aspect_ratio": 2,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 6,
      "shape_type": "triangle",
      "name": "triangle",
      "shape_aspect_ratio": 1,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 7,
      "shape_type": "triangle",
      "name": "wide_triangle",
      "shape_aspect_ratio": 0.5,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 8,
      "shape_type": "triangle",
      "name": "narrow_triangle",
      "shape_aspect_ratio": 2,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    },
    {
      "id": 9,
      "shape_type": "trapezoid",
      "name": "trapezoid",
      "shape_aspect_ratio": 2,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ],
      "sides": 5
    },
    {
      "id": 10,
      "shape_type": "hexagon",
      "name": "hexagon",
      "shape_aspect_ratio": 2,
      "fill_color": [
        120,
        0,
        200
      ],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ],
      "sides": 6
    },
    {
      "id": 11,
      "shape_type": "rectangle",
      "name": "narrow_rectangle_frame",
      "shape_aspect_ratio": 2,
      "fill_color": [],
      "outline_color": [
        120,
        0,
        200
      ],
      "shape_width_choices": [
        30,
        60,
        100
      ]
    }
  ]
}
```

**Anottations File**

The annotation files are also generated by the create shape procedure, a fle per each created section, i.e. train, test, validation.
Each file's row hold an image file path with its corresponding shapes classes and bounding boxes.

**Installation**

Clone the repo:

`git clone https://github.com/ronen-halevy/shapes-dataset.git`

Install Required Packages:

`pip install -r requirements.txt`

**Generate A Dataset**

(The repo's current produced dataset is based on the commited json files).

Generation Execution:

`python create_shapes_dataset.py config.json shapes.json`


**Display Example Images Plots With Annotations**

`python plot_images.py config.json shapes.json [section]`

Where section is the set to pick images from, which can be `train' (default), 'test', or valid'.

An example plot follows:
![alt text](https://github.com/ronen-halevy/shapes-dataset/blob/main/docs/shape_boxes-1.png)



![alt text](https://github.com/ronen-halevy/shapes-dataset/blob/main/docs/shape_boxes-2.png)





