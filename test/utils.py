from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL.Image import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np


# from test

def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    """
    Description: Draws a text which starts at xmin,ymin bbox corner

    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                  font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_margin_factor = 0.05

    left, top, right, bottom = zip(*[font.getbbox(display_str) for display_str in display_str_list])
    text_heights = tuple(map(lambda i, j: i - j, bottom, top))
    text_widths = tuple(map(lambda i, j: i - j, right, left))

    text_margins = np.ceil(text_margin_factor * np.array(text_heights))
    text_bottoms = ymin * (ymin > text_heights) + (ymin + text_heights) * (ymin <= text_heights)

    for idx, (display_str, xmint, text_bottom, text_width, text_height, text_margin) in enumerate(
            zip(display_str_list, xmin, text_bottoms, text_widths, text_heights, text_margins)):
        left, top, right, bottom = font.getbbox(display_str)
        text_height = bottom - top
        text_width = right - left

        text_margin = np.ceil(text_margin_factor * text_height)

        draw.rectangle(((xmint, text_bottom - text_height - 2 * text_margin),
                        (xmint + text_width + text_margin, text_bottom)),
                       fill=tuple(color))

        draw.text((xmint + text_margin, text_bottom - text_height - 3 * text_margin),
                  display_str,
                  fill="black",
                  font=font)
    return image


def draw_bounding_box(image, boxes, thickness=1):
    colors = list(ImageColor.colormap.values())
    color = colors[0]
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, w, h = box
        # xmin = xc-w/2
        # ymin = yc-h/2
        print((xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin))
        draw.line([(xmin, ymin), (xmin, ymin + h), (xmin + w, ymin + h), (xmin + w, ymin),
                   (xmin, ymin)],
                  width=thickness,
                  fill=color)
    return image

def draw_dataset_entry(image, bboxes, category_names, title, output_path):
    annotated_bbox_image = draw_bounding_box(image, bboxes)
    text_box_color = [255, 255, 255]
    draw_text_on_bounding_box(annotated_bbox_image, np.array(bboxes)[..., 1],
                                                     np.array(bboxes)[..., 0], text_box_color,
                                                     category_names, font_size=15)

    print(f'saving test results to {output_path}')
    image.save(output_path)

