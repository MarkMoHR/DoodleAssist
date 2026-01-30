import numpy as np
from svgpathtools.parser import parse_transform
import xml.etree.ElementTree as ET
from svg.path import parse_path, path

invalid_svg_shapes = ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']


def parse_single_path(path_str):
    ps = parse_path(path_str)

    mul_paths_list = []
    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)

        if path_type == path.Move:
            # assert item_i == 0
            if len(control_points_list) > 0:
                assert len(control_points_list) > 1
                mul_paths_list.append(control_points_list)
                control_points_list = []

            start = path_item.start
            start_x, start_y = start.real, start.imag
            control_points_list.append((start_x, start_y))
        elif path_type == path.CubicBezier:
            start, control1, control2, end = path_item.start, path_item.control1, path_item.control2, path_item.end
            start_x, start_y = start.real, start.imag
            control1_x, control1_y = control1.real, control1.imag
            control2_x, control2_y = control2.real, control2.imag
            end_x, end_y = end.real, end.imag
            # control_points_list.append((control1_x, control1_y))
            # control_points_list.append((control2_x, control2_y))
            control_points_list.append((end_x, end_y))
        elif path_type == path.Arc:
            raise Exception('Arc is here')
        elif path_type == path.Line:
            start, end = path_item.start, path_item.end
            start_x, start_y = start.real, start.imag
            end_x, end_y = end.real, end.imag
            control_points_list.append((end_x, end_y))
        elif path_type == path.Close:
            assert item_i == len(ps) - 1
        else:
            raise Exception('Unknown path_type', path_type)

    mul_paths_list.append(control_points_list)
    assert len(mul_paths_list) >= 1

    return mul_paths_list


def matrix_transform(points, matrix_params):
    # points: (N, 2), (x, y)
    # matrix_params: (6)
    new_points = []
    a, b, c, d, e, f = matrix_params
    matrix = np.array([[a, b, c],
                       [d, e, f],
                       [0, 0, 1]], dtype=np.float32)
    for point in points:
        point_vec = [point[0], point[1], 1]
        new_point = np.matmul(matrix, point_vec)[:2]
        new_points.append(new_point)
        # print(point, new_point)
    new_points = np.stack(new_points).astype(np.float32)
    return new_points


def matrix_transform_path(control_points_single_paths_, transform_mat_params):
    control_points_single_paths_out = []  # list of (N_point, 2)
    for single_path in control_points_single_paths_:  # (N_point, 2)
        new_path = matrix_transform(single_path, transform_mat_params).tolist()
        control_points_single_paths_out.append(new_path)
    return control_points_single_paths_out


def parse_svg(svg_file, is_merge=False):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    width = root.get('width')
    height = root.get('height')
    width = int(width)
    height = int(height)

    view_box = root.get('viewBox')
    if view_box is not None:
        view_x, view_y, view_width, view_height = view_box.split(' ')
        view_x, view_y, view_width, view_height = int(view_x), int(view_y), int(view_width), int(view_height)
        assert view_x == 0 and view_y == 0
        assert width == view_width and height == view_height

    paths_list = []
    ids_list = []
    transformed_ids_list = []

    group_transform_params = None
    group_transform_count = 0

    for ei, elem in enumerate(root.iter()):
        try:
            _, tag_suffix = elem.tag.split('}')
        except ValueError:
            continue

        if tag_suffix == 'g':
            if 'transform' in elem.attrib.keys():
                transform_str = elem.attrib['transform']
                transform_mat = parse_transform(transform_str)
                group_transform_params = np.reshape(transform_mat, (9))[:6]
                sub_elems = elem.findall('{http://www.w3.org/2000/svg}path')
                group_transform_count = len(sub_elems)
        assert tag_suffix not in invalid_svg_shapes

        if tag_suffix == 'path':
            path_d = elem.attrib['d']
            assert 'id' in elem.attrib.keys()
            path_id = elem.attrib['id']
            ids_list.append(path_id)
            control_points_single_paths = parse_single_path(path_d)  # list of (N_point, 2)

            ## self-transform
            if 'transform' in elem.attrib.keys():
                transform_str = elem.attrib['transform']
                transform_mat = parse_transform(transform_str)
                transform_mat_params = np.reshape(transform_mat, (9))[:6]
                control_points_single_paths = matrix_transform_path(control_points_single_paths, transform_mat_params)
                transformed_ids_list.append(path_id)

            ## group-transform
            if group_transform_count > 0:
                assert group_transform_params is not None
                control_points_single_paths = matrix_transform_path(control_points_single_paths, group_transform_params)
                group_transform_count -= 1
                transformed_ids_list.append(path_id)

            if is_merge:
                paths_list.append(control_points_single_paths)
            else:
                paths_list += control_points_single_paths

    assert len(paths_list) > 0
    return (width, height), paths_list, ids_list, transformed_ids_list


def parse_svg_path_count(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    path_count = 0
    for ei, elem in enumerate(root.iter()):
        try:
            _, tag_suffix = elem.tag.split('}')
        except ValueError:
            continue

        if tag_suffix == 'path':
            path_count += 1

    assert path_count > 0
    return path_count
