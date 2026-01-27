# import gizeh
from PIL import Image
import numpy as np
import cairocffi as cairo


# def rasterize_line_image(paths_list_norm, raster_size, stroke_width, background_img_path=None, save_path=None):
#     '''
#     paths_list_norm: list of [(N_points, 2), ...]
#     '''
#     line_groups = []
#     surface = gizeh.Surface(width=raster_size, height=raster_size, bg_color=(1, 1, 1))  # in pixels
#     if background_img_path is not None:
#         light_color_factor = 0.3
#
#         background_img = Image.open(background_img_path).convert('L')
#         background_img = np.array(background_img, dtype=np.float32)
#         background_img = np.stack([background_img, 255.0 - (255.0 - background_img) * 0.5, np.ones_like(background_img) * 255.0], axis=-1)
#
#         background_img = 255.0 - (255.0 - background_img) * light_color_factor
#         background_img = background_img.astype(np.uint8)
#
#         surface = gizeh.Surface(width=raster_size, height=raster_size, bg_color=(1, 1, 1)).from_image(background_img)
#     for cluster_idx in range(len(paths_list_norm)):
#         path_cluster = paths_list_norm[cluster_idx]
#         assert type(path_cluster) is list
#         assert len(path_cluster) >= 1
#
#         stroke_color = (0, 0, 0)
#         for path_points_resize in path_cluster:
#             if background_img_path is not None:
#                 line = gizeh.polyline(points=path_points_resize, stroke_width=stroke_width * 3, stroke=stroke_color)
#             else:
#                 line = gizeh.polyline(points=path_points_resize, stroke_width=stroke_width, stroke=stroke_color)
#             line_groups.append(line)
#
#     group_i = gizeh.Group(line_groups)
#     group_i.draw(surface)
#
#     result_img = surface.get_npimage()  # returns a (width x height x 3) numpy array
#     if save_path is not None:
#         surface.write_to_png(save_path)
#     return result_img


def rasterize_line_image(paths_list_norm, raster_size, stroke_width,
                         background_img_path=None, save_path=None,
                         bg_color=(1, 1, 1), stroke_color=(0, 0, 0), light_color_factor=0.3):
    '''
    paths_list_norm: list of [(N_points, 2), ...]
    '''
    if background_img_path is not None:
        background_img = Image.open(background_img_path).convert('L')
        background_img = np.array(background_img, dtype=np.float32)
        background_img = np.stack(
            [background_img, 255.0 - (255.0 - background_img) * 0.5, np.ones_like(background_img) * 255.0,
             np.ones_like(background_img) * 255.0], axis=-1)

        background_img = 255.0 - (255.0 - background_img) * light_color_factor
        background_img = background_img.astype(np.uint8)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, raster_size, raster_size, data=background_img)
    else:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, raster_size, raster_size)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(stroke_width)

    if background_img_path is None:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()

    ctx.set_source_rgb(*stroke_color)
    for cluster_idx in range(len(paths_list_norm)):
        path_cluster = paths_list_norm[cluster_idx]  # [(N_points, 2), ...]
        assert type(path_cluster) is list
        assert len(path_cluster) >= 1

        for path_points_resize in path_cluster:  # (N_points, 2)
            assert len(path_points_resize) > 0
            for si in range(len(path_points_resize) - 1):
                x0, y0 = path_points_resize[si]
                x1, y1 = path_points_resize[si + 1]
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
                ctx.stroke()

    surface_data = surface.get_data()
    result_img = np.copy(np.asarray(surface_data)).reshape(raster_size, raster_size, 4)[:, :, :3]  # (width x height x 3)

    if save_path is not None:
        result_img_png = Image.fromarray(result_img, 'RGB')
        result_img_png.save(save_path, 'PNG')
    return result_img
