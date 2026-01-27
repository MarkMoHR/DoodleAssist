from PIL import Image
import numpy as np
import cv2

import alphashape
from shapely.geometry import multipolygon, polygon, point

from rasterize_utils import rasterize_line_image


def boundary_poly2mask(img_path, save_mask_path, shrink_factor=0.01):
    '''
    From existing image
    '''
    sketch_img = Image.open(img_path).convert('L')
    h, w = sketch_img.height, sketch_img.width
    sketch_img = np.array(sketch_img, dtype=np.float32)
    sketch_img[sketch_img < 128] = 0
    sketch_img[sketch_img != 0] = 255

    stroke_pixels = np.argwhere(sketch_img == 0)  # (N, 2), (y, x)

    all_polygons = []
    shrink_factor_curr = shrink_factor
    while True:
        alpha_shape_out = alphashape.alphashape(stroke_pixels, shrink_factor_curr)
        if type(alpha_shape_out) is multipolygon.MultiPolygon:
            shrink_factor_curr /= 2.0
            print('shrink_factor_curr', shrink_factor_curr)
            # raise Exception('MultiPolygon is not allowed here')
            # for polygon_ in alpha_shape_out.geoms:
            #     all_polygons.append(polygon_)
        elif type(alpha_shape_out) is polygon.Polygon:
            all_polygons.append(alpha_shape_out)
            break
        else:
            raise Exception('Unknown type:', type(alpha_shape_out))
    assert len(all_polygons) == 1

    all_coords_list = []
    for polygon_ in all_polygons:
        line_string = polygon_.boundary
        vertices = line_string.coords

        vertices_list = []
        for vertex in vertices:
            vertices_list.append(vertex)
        vertices_list = np.array(vertices_list, dtype=np.int32)
        vertices_list = vertices_list[:, ::-1]
        all_coords_list.append(vertices_list)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, all_coords_list, -1, 255, -1)
    cv2.imwrite(save_mask_path, mask)
    return mask


def boundary_poly2mask_v2(path_params, raster_size, stroke_width, save_mask_path=None, shrink_factor=0.01):
    '''
    From path parameters; for single image
      path_params: list of [(N_points, 2), ...]
    '''
    if len(path_params) == 0:
        return np.zeros((raster_size, raster_size), dtype=np.uint8)

    sketch_img = rasterize_line_image(path_params, raster_size, stroke_width)
    # numpy.ndarray, (W, H, 3), [0-stroke, 255-BG], uint8

    h, w = sketch_img.shape[1], sketch_img.shape[0]
    sketch_img = np.array(sketch_img[:, :, 0], dtype=np.float32)
    sketch_img[sketch_img < 128] = 0
    sketch_img[sketch_img != 0] = 255

    stroke_pixels = np.argwhere(sketch_img == 0)  # (N, 2), (y, x)

    all_polygons = []
    shrink_factor_curr = shrink_factor
    while True:
        alpha_shape_out = alphashape.alphashape(stroke_pixels, shrink_factor_curr)
        if type(alpha_shape_out) is multipolygon.MultiPolygon:
            shrink_factor_curr /= 2.0
            # print('shrink_factor_curr', shrink_factor_curr)
            # raise Exception('MultiPolygon is not allowed here')
            # for polygon_ in alpha_shape_out.geoms:
            #     all_polygons.append(polygon_)
        elif type(alpha_shape_out) is polygon.Polygon:
            all_polygons.append(alpha_shape_out)
            break
        elif type(alpha_shape_out) is point.Point:
            break
        else:
            raise Exception('Unknown type:', type(alpha_shape_out))
    assert len(all_polygons) <= 1

    all_coords_list = []
    for polygon_ in all_polygons:
        line_string = polygon_.boundary
        vertices = line_string.coords

        vertices_list = []
        for vertex in vertices:
            vertices_list.append(vertex)
        vertices_list = np.array(vertices_list, dtype=np.int32)
        vertices_list = vertices_list[:, ::-1]
        all_coords_list.append(vertices_list)

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(all_coords_list) > 0:
        cv2.drawContours(mask, all_coords_list, -1, 255, -1)
    if save_mask_path is not None:
        cv2.imwrite(save_mask_path, mask)
    return mask
