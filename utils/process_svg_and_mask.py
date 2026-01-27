import sys
sys.path.append('./utils')

import os
import numpy as np
from PIL import Image
import cv2

from svg_utils import parse_svg
from mask_utils import boundary_poly2mask_v2
from rasterize_utils import rasterize_line_image
from modify_mode_utils import detect_mode, ModifyMode


def normalize_path_coords(paths_list, view_sizes, raster_size):
    points_scaling = max(view_sizes) / float(raster_size)
    paths_list_norm = []  # list of [(N_points, 2), ...]
    for pi, path_cluster in enumerate(paths_list):
        assert type(path_cluster) is list, type(path_cluster)
        assert len(path_cluster) >= 1

        path_cluster_norm = []
        for path_points in path_cluster:
            path_points_resize = np.array(path_points, dtype=np.float32) / points_scaling
            path_cluster_norm.append(path_points_resize)
        # path_cluster_norm = np.stack(path_cluster_norm, axis=0)
        paths_list_norm.append(path_cluster_norm)
    return paths_list_norm


def get_modified_path_idx(original_paths, modified_paths, detected_modified_path_ids, path_diff_threshold=0.5):
    modified_path_ids = {'add': [], 'edit': [], 'delete': []}
    path_ids_ori = original_paths.keys()
    path_ids_new = modified_paths.keys()
    add_path_ids = set(path_ids_new).difference(set(path_ids_ori))
    delete_path_ids = set(path_ids_ori).difference(set(path_ids_new))
    modified_path_ids['add'] += list(add_path_ids)
    modified_path_ids['delete'] += list(delete_path_ids)
    modified_path_ids['edit'] += detected_modified_path_ids

    for path_id in path_ids_ori:
        if path_id in modified_path_ids['add'] or path_id in modified_path_ids['edit'] or path_id in modified_path_ids['delete']:
            continue

        assert path_id in path_ids_new
        ori_paths_list = original_paths[path_id]  # [(N_points, 2), ...]
        new_paths_list = modified_paths[path_id]  # [(N_points, 2), ...]
        if len(ori_paths_list) != len(new_paths_list):
            modified_path_ids['edit'].append(path_id)
            continue

        for i in range(len(ori_paths_list)):
            ori_path = ori_paths_list[i]  # (N_points, 2)
            new_path = new_paths_list[i]  # (N_points, 2)
            if ori_path.shape[0] != new_path.shape[0]:
                modified_path_ids['edit'].append(path_id)
                break

            path_point_dist = np.sqrt(np.sum(np.power(ori_path - new_path, 2), axis=-1))  # (N_points)
            path_point_dist_max = np.max(path_point_dist)
            if path_point_dist_max >= path_diff_threshold:
                modified_path_ids['edit'].append(path_id)
                break

    return modified_path_ids


def gen_mask_vis(lineart_file, mask_, light_factor=0.15, save_path=None):
    # mask: (H, W), [0-BG, 255-FG], uint8
    lineart_img = Image.open(lineart_file).convert('RGB')
    lineart_img = np.array(lineart_img, dtype=np.float32)  # [0-stroke, 255-BG]

    ## red
    color_mask = np.zeros_like(lineart_img)
    color_mask[:, :, 0].fill(255)

    ## gray
    # color_mask = np.zeros_like(lineart_img)
    # color_mask.fill(52)

    if type(mask_) is str:
        mask = Image.open(mask_).convert('L')
        mask = np.array(mask, dtype=np.uint8)
    elif mask_ is None:
        mask = np.zeros_like(color_mask)[:, :, 0]
    else:
        mask = np.copy(mask_)
    mask_bin = mask.astype(np.float32) / 255.0 * light_factor  # [0-BG, 1-FG]
    mask_bin = np.stack([mask_bin] * 3, axis=-1)
    lineart_img_masked = lineart_img * (1 - mask_bin) + color_mask * mask_bin
    lineart_img_masked = lineart_img_masked.astype(np.uint8)  # (H, W, 3), uint8
    if save_path is not None:
        lineart_img_masked_png = Image.fromarray(lineart_img_masked, 'RGB')
        lineart_img_masked_png.save(save_path, 'PNG')
    return lineart_img_masked


def gen_sketch_vis(data_base, modified_svg_name, previous_lineart_file,
                   raster_size=512, stroke_width=0.8):
    modified_file = os.path.join(data_base, modified_svg_name)
    save_path_overlay = modified_file[:-4] + '_overlay.png'
    new_view_sizes, new_paths_list, _, _ = parse_svg(modified_file, is_merge=True)  # list of [(N_points, 2), ...]
    new_paths_list_norm = normalize_path_coords(new_paths_list, new_view_sizes, raster_size)  # list of [(N_points, 2), ...]
    return rasterize_line_image(new_paths_list_norm, raster_size, stroke_width, background_img_path=previous_lineart_file,
                                save_path=save_path_overlay)


def maskA_included_by_maskB(mask_a, mask_b, pixel_percentage_threshold=0.05, pixel_num_threshold=25):
    # mask_a, mask_b: numpy.ndarray, (H, W), [0-BG, 255-FG], uint8
    mask_diff = np.copy(mask_a).astype(np.float32) - np.copy(mask_b).astype(np.float32)
    mask_diff = np.clip(mask_diff, 0.0, 255.0).astype(np.uint8)
    return np.sum(mask_diff) <= np.sum(mask_b) * pixel_percentage_threshold

    # mask_diff[mask_diff < 128] = 0
    # mask_diff[mask_diff != 0] = 255
    # diff_pixel = np.sum(mask_diff) / 255.0
    # return diff_pixel <= pixel_num_threshold


def mask_add_mode(modified_path_ids, new_paths, ori_paths_list_norm, new_paths_list_norm,
                  raster_size, stroke_width, shrink_factor,
                  kernel_l, kernel_s):
    added_paths_ids = modified_path_ids['add']
    added_paths = [new_paths[added_paths_id] for added_paths_id in added_paths_ids]  # list of [(N_points, 2), ...]

    previous_mask = boundary_poly2mask_v2(ori_paths_list_norm, raster_size, stroke_width, shrink_factor=shrink_factor)
    new_mask = boundary_poly2mask_v2(new_paths_list_norm, raster_size, stroke_width, shrink_factor=shrink_factor)
    new_stroke_mask = boundary_poly2mask_v2(added_paths, raster_size, stroke_width, shrink_factor=shrink_factor)
    # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8

    if maskA_included_by_maskB(new_mask, previous_mask):
        added_mask = cv2.dilate(new_stroke_mask, kernel_l)  # (H, W)
    else:
        # added_mask = cv2.dilate(new_mask, kernel_l).astype(np.float32) - cv2.dilate(previous_mask, kernel_s).astype(np.float32)
        added_mask = cv2.dilate(
            np.clip(cv2.dilate(new_mask, kernel_s).astype(np.float32) - cv2.dilate(previous_mask, kernel_l).astype(np.float32),
                    0.0, 255.0).astype(np.uint8),
            kernel_s)
        if np.sum(added_mask) == 0:
            added_mask = cv2.dilate(new_stroke_mask, kernel_l)  # (H, W)
    return added_mask


def mask_edit_mode(modified_path_ids, ori_paths, new_paths,
                   raster_size, stroke_width, shrink_factor,
                   kernel):
    # editing strokes only: use region before and after
    edited_mask = np.zeros((raster_size, raster_size), dtype=np.float32)  # [0-BG, 255-FG]

    edited_paths_ids = modified_path_ids['edit']
    edited_ori_paths = [ori_paths[edited_paths_id] for edited_paths_id in edited_paths_ids]  # list of [(N_points, 2), ...]
    edited_new_paths = [new_paths[edited_paths_id] for edited_paths_id in edited_paths_ids]

    ori_mask = boundary_poly2mask_v2(edited_ori_paths, raster_size, stroke_width, shrink_factor=shrink_factor)
    new_mask = boundary_poly2mask_v2(edited_new_paths, raster_size, stroke_width, shrink_factor=shrink_factor)
    # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8

    edited_mask += ori_mask.astype(np.float32)
    edited_mask += new_mask.astype(np.float32)

    edited_mask = np.clip(edited_mask, 0.0, 255.0)
    edited_mask = edited_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    edited_mask = cv2.dilate(edited_mask, kernel)  # (H, W)
    return edited_mask


def mask_delete_mode(modified_path_ids, ori_paths,
                     raster_size, stroke_width, shrink_factor,
                     kernel):
    # deleting strokes only: use region before
    deleted_paths_ids = modified_path_ids['delete']
    deleted_ori_paths = [ori_paths[deleted_paths_id] for deleted_paths_id in deleted_paths_ids]  # list of [(N_points, 2), ...]

    deleted_mask = boundary_poly2mask_v2(deleted_ori_paths, raster_size, stroke_width, shrink_factor=shrink_factor)
    # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8
    deleted_mask = cv2.dilate(deleted_mask, kernel)  # (H, W)
    return deleted_mask


def read_modified_svg_and_produce_mask(data_base, modified_svg_name,
                                       previous_lineart_file, previous_svg_name,
                                       mask_dilate_size=1, save_mask_vis=True,
                                       raster_size=512, stroke_width=0.8, shrink_factor=0.01):
    ori_file = None if previous_svg_name is None else os.path.join(data_base, previous_svg_name)
    modified_file = os.path.join(data_base, modified_svg_name)

    ## 1. load original and new sketch data
    ori_ids_list = []
    ori_paths_list_norm = []
    if ori_file is not None:
        ori_view_sizes, ori_paths_list, ori_ids_list, _ = parse_svg(ori_file, is_merge=True)  # list of [(N_points, 2), ...]
        ori_paths_list_norm = normalize_path_coords(ori_paths_list, ori_view_sizes, raster_size)  # list of [(N_points, 2), ...]
    ori_paths = {}
    for i, ori_id in enumerate(ori_ids_list):
        ori_paths[ori_id] = ori_paths_list_norm[i]

    new_view_sizes, new_paths_list, new_ids_list, transformed_ids_list_part = parse_svg(modified_file, is_merge=True)  # list of [(N_points, 2), ...]
    new_paths_list_norm = normalize_path_coords(new_paths_list, new_view_sizes, raster_size)  # list of [(N_points, 2), ...]
    new_paths = {}
    for i, new_id in enumerate(new_ids_list):
        new_paths[new_id] = new_paths_list_norm[i]

    transformed_ids_list_part = []

    ## 2. calculate modified line art
    save_path = modified_file[:-4] + '.png'
    save_path_overlay = modified_file[:-4] + '_overlay.png'
    rasterize_line_image(new_paths_list_norm, raster_size, stroke_width, background_img_path=None, save_path=save_path)
    rasterize_line_image(new_paths_list_norm, raster_size, stroke_width, background_img_path=previous_lineart_file, save_path=save_path_overlay)

    print('-' * 20)
    modified_path_ids = get_modified_path_idx(ori_paths, new_paths, transformed_ids_list_part)
    print('modified_path_ids', modified_path_ids)
    detected_modified_mode = detect_mode(modified_path_ids)
    print('detected_modified_mode:', ModifyMode.mode_dict[detected_modified_mode])

    ## 3. calculate modified region mask
    modified_mask = np.zeros((raster_size, raster_size), dtype=np.float32)  # [0-BG, 255-FG]
    kernel_l = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_dilate_size, mask_dilate_size))
    kernel_s = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_dilate_size // 2, mask_dilate_size // 2))

    if detected_modified_mode == ModifyMode.EDIT_ONLY:
        modified_mask = mask_edit_mode(modified_path_ids, ori_paths, new_paths,
                                       raster_size, stroke_width, shrink_factor,
                                       kernel_l)

    elif detected_modified_mode == ModifyMode.DELETE_ONLY:
        modified_mask = mask_delete_mode(modified_path_ids, ori_paths,
                                         raster_size, stroke_width, shrink_factor,
                                         kernel_l)

    elif detected_modified_mode == ModifyMode.ADD_ONLY:
        modified_mask = mask_add_mode(modified_path_ids, new_paths, ori_paths_list_norm, new_paths_list_norm,
                                      raster_size, stroke_width, shrink_factor,
                                      kernel_l, kernel_s)  # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8

    elif detected_modified_mode == ModifyMode.ADD_EDIT:
        ## edit
        edited_mask = mask_edit_mode(modified_path_ids, ori_paths, new_paths,
                                     raster_size, stroke_width, shrink_factor,
                                     kernel_l)
        modified_mask += edited_mask.astype(np.float32)

        ## add
        added_mask = mask_add_mode(modified_path_ids, new_paths, ori_paths_list_norm, new_paths_list_norm,
                                   raster_size, stroke_width, shrink_factor,
                                   kernel_l, kernel_s)  # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8
        modified_mask += added_mask.astype(np.float32)

        ## summary
        modified_mask = np.clip(modified_mask, 0.0, 255.0)
        modified_mask = modified_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    elif detected_modified_mode == ModifyMode.ADD_DELETE:
        ## delete first
        deleted_mask = mask_delete_mode(modified_path_ids, ori_paths,
                                        raster_size, stroke_width, shrink_factor,
                                        kernel_l)
        modified_mask += deleted_mask.astype(np.float32)

        ## add then
        added_mask = mask_add_mode(modified_path_ids, new_paths, ori_paths_list_norm, new_paths_list_norm,
                                   raster_size, stroke_width, shrink_factor,
                                   kernel_l, kernel_s)  # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8
        modified_mask += added_mask.astype(np.float32)

        ## summary
        modified_mask = np.clip(modified_mask, 0.0, 255.0)
        modified_mask = modified_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    elif detected_modified_mode == ModifyMode.EDIT_DELETE:
        ## delete
        deleted_mask = mask_delete_mode(modified_path_ids, ori_paths,
                                        raster_size, stroke_width, shrink_factor,
                                        kernel_l)
        modified_mask += deleted_mask.astype(np.float32)

        ## edit
        edited_mask = mask_edit_mode(modified_path_ids, ori_paths, new_paths,
                                     raster_size, stroke_width, shrink_factor,
                                     kernel_l)
        modified_mask += edited_mask.astype(np.float32)

        ## summary
        modified_mask = np.clip(modified_mask, 0.0, 255.0)
        modified_mask = modified_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    elif detected_modified_mode == ModifyMode.ADD_EDIT_DELETE:
        ## delete
        deleted_mask = mask_delete_mode(modified_path_ids, ori_paths,
                                        raster_size, stroke_width, shrink_factor,
                                        kernel_l)
        modified_mask += deleted_mask.astype(np.float32)

        ## edit
        edited_mask = mask_edit_mode(modified_path_ids, ori_paths, new_paths,
                                     raster_size, stroke_width, shrink_factor,
                                     kernel_l)
        modified_mask += edited_mask.astype(np.float32)

        ## add then
        added_mask = mask_add_mode(modified_path_ids, new_paths, ori_paths_list_norm, new_paths_list_norm,
                                   raster_size, stroke_width, shrink_factor,
                                   kernel_l, kernel_s)  # numpy.ndarray, (H, W), [0-BG, 255-FG], uint8
        modified_mask += added_mask.astype(np.float32)

        ## summary
        modified_mask = np.clip(modified_mask, 0.0, 255.0)
        modified_mask = modified_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    elif detected_modified_mode == ModifyMode.UNCHANGED:
        modified_mask = modified_mask.astype(np.uint8)  # (H, W), [0-BG, 255-FG], uint8

    else:
        raise Exception('Invalid detected_modified_mode')

    ## visualize mask
    if save_mask_vis:
        modified_mask_img = Image.fromarray(modified_mask, 'L')
        mask_save_path = os.path.join(data_base, modified_svg_name[:-4] + '-mask.png')
        modified_mask_img.save(mask_save_path, 'PNG')

        vis_mask_save_path = os.path.join(data_base, modified_svg_name[:-4] + '-mask_vis.png')
    else:
        vis_mask_save_path = None
    previous_lineart_img_masked = gen_mask_vis(previous_lineart_file, modified_mask, save_path=vis_mask_save_path)

    return modified_mask, previous_lineart_img_masked
