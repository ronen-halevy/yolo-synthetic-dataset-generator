import numpy as np
import math
from src.create.create_polygons import CreatePolygons


class CreateSegmentationEntries(CreatePolygons):
    def __init__(self, config):
        CreatePolygons.__init__(self, config)
    def run(self, nentries):
        batch_image_size, batch_categories_ids, batch_categories_names, batch_polygons, batch_objects_colors, batch_obb_thetas = self.create_batch_polygons(
            nentries)
        batch_labels = self.arrange_segmentation_entries(batch_polygons, batch_image_size, batch_categories_ids)
        return batch_polygons, batch_labels, batch_objects_colors, batch_image_size

    def arrange_segmentation_entries(self, images_polygons, images_size, categories_lists):
        batch_entries = []
        for image_polygons, image_size, class_ids in zip(images_polygons, images_size,
                                                         categories_lists):
            # normalize sizes:
            image_polygons = [image_polygon / np.array(image_size) for image_polygon in image_polygons]

            image_entries = [
                f"{category_id} {' '.join(str(vertix) for vertix in list(image_polygon.reshape(-1)))}\n" for
                image_polygon, category_id in zip(image_polygons, class_ids)]
            batch_entries.append(image_entries)
        return batch_entries
