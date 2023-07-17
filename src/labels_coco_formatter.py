import json
from datetime import date, datetime


# Create a coco like format label file. format:
# a single json file with 4 tables:
#     "info":
#     "licenses":
#     "images": images_records,
#     "categories": categories_records,
#     "annotations": annotatons_records
def coco_formatter(images_paths, images_sizes, images_bboxes, images_objects_categories_indices,
                   category_names, super_category_names, annotations_output_path):
    """
     :param images_paths: list of dataset image filenames
    :param images_sizes: list of per image [im.height, im.width] tuples
    :param images_bboxes: list of per image bboxes arrays in xyxy format.
    :param images_objects_categories_indices: list of per image categories_indices arrays
    :param category_names: list of all dataset's category names
    :param super_category_names:  list of all dataset's super_category_names
    :param annotations_output_path: path for output file storage
    :return:
    """

    print('create_coco_labels_file')

    anno_id = 0
    # for example_id in range(nex):
    added_category_names = []
    categories_records = []
    # map_categories_id = {}

    # fill category
    id = 0
    for category_name, supercategory in zip(category_names, super_category_names):

        if category_name not in added_category_names:
            categories_records.append({
                "supercategory": supercategory,
                "id": id,
                "name": category_name,
            })
            added_category_names.append(category_name)
            # map_categories_id.update({category_name: id})
            id += 1

    images_records = []
    annotatons_records = []
    for example_id, (image_path, image_size, bboxes, objects_categories_indices) in enumerate(
            zip(images_paths, images_sizes, images_bboxes, images_objects_categories_indices)):

        # images records:

        images_records.append({
            "license": '',
            "file_name": image_path,
            "coco_url": "",
            'width': image_size[1],
            'height': image_size[0],
            "date_captured": str(datetime.now()),
            "flickr_url": "",
            "id": example_id
        })

        # annotatons_records
        for bbox, category_id in zip(bboxes, objects_categories_indices):
            annotatons_records.append({
                "segmentation": [],
                "area": [],
                "iscrowd": 0,
                "image_id": example_id,
                "bbox": list(bbox),
                "category_id": category_id,
                "id": anno_id
            }
            )
            anno_id += 1
    date_today = date.today()
    info = {
        "description": " Dataset",
        "url": '',
        # "version": config.get('version', 1.0),
        "year": date_today.year,
        "contributor": '',
        "date_created": str(date_today),
        "licenses": '',
        "categories": categories_records
    }
    output_records = {
        "info": info,
        "licenses": [],
        "images": images_records,
        "categories": categories_records,
        "annotations": annotatons_records
    }
    print(f'Save annotation  in {annotations_output_path}')
    with open(annotations_output_path, 'w') as fp:
        json.dump(output_records, fp)
