import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from matplotlib import pyplot
import cv2

from .transform_utils import bdb3d_from_front_face, IGTransform, interpolate_line


def plot_layout(layout, cameras=None):
    def plot_poly(poly):
        pyplot.plot(*poly.exterior.xy)
        for i in poly.interiors:
            pyplot.plot(*i.xy)

    if isinstance(layout, Polygon):
        plot_poly(layout)
    else:
        for poly in layout:
            plot_poly(poly)

    def plot_camera(camera):
        pos = camera['pos'][:2]
        target = camera['target'][:2]
        pyplot.arrow(*pos, target[0] - pos[0], target[1] - pos[1], width=0.1)
        pyplot.text(*pos, camera['name'], fontsize=12)

    if cameras:
        if isinstance(cameras, dict):
            plot_camera(cameras)
        else:
            for camera in cameras:
                plot_camera(camera)

    pyplot.axis('equal')
    pyplot.show()

def scene_layout_from_scene(scene):
    # show_image(scene.room_sem_map)
    room_ids = np.unique(scene.room_sem_map).tolist()
    room_ids.remove(0)
    rooms = []
    for i_room in room_ids:
        room_mask = scene.room_sem_map == i_room
        contours, hierarchy = cv2.findContours(
            room_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i_max = np.argmax([cv2.contourArea(contour) for contour in contours])
        contour = contours[i_max][:, 0, :]
        room = scene.seg_map_to_world(contour[:, ::-1])
        rooms.append(Polygon(room))
    # plot_layout(rooms)
    return {'rooms': rooms}


def room_layout_from_scene_layout(camera, scene_layout):
    # find which room in
    camera_point = Point(*camera['pos'][:2])
    in_wall = True
    for room_layout in scene_layout['rooms']:
        if room_layout.contains(camera_point):
            # plot_layout(r, camera)
            # sort boundary points in clockwise order
            room_layout = shapely.geometry.polygon.orient(room_layout, -1)
            boundary = np.array(room_layout.boundary.xy)[:, :-1].T
            assert np.all(np.linalg.norm((np.roll(boundary, 1, 0) - boundary), ord=2, axis=1) > 0.01), \
                'neighboring corners are too close'
            in_wall = False
            break
    if in_wall:
        print("in wall")
        return

    nearest_point, _ = shapely.ops.nearest_points(room_layout.boundary, camera_point)
    distance_wall = camera_point.distance(nearest_point)
    if distance_wall < 0.5:
        print(f"too close ({distance_wall:.3f} < 0.5) to wall")
        return
    if not room_layout.is_valid:
        print("not valid")
        return

    return {'room': room_layout, 'height': scene_layout['height']}


def manhattan_pix_layout_from_room_layout(camera, room_layout):
    if room_layout is None:
        return

    # generate pano Manhattan layout from room layout
    height, width = camera['height'], camera['width']
    boundary = np.array(room_layout['room'].boundary.xy)[:, :-1].T
    camera_height = camera['pos'][-1]
    directions = []
    camera_point = np.array(camera['pos'][:2])
    front = np.arctan2(*(np.array(camera['target'][:2]) - camera_point))
    points = []
    for p in boundary:
        direction = np.mod(np.arctan2(*(p - camera_point)) - front + np.pi, np.pi * 2)
        directions.append(direction)
        x = direction / (2 * np.pi) * width
        dis = np.linalg.norm(p - camera_point, 2)
        pitch = np.arctan2(room_layout['height'] - camera_height, dis)
        y_top = (np.pi / 2 - pitch) / np.pi * height
        points.append([x, y_top])
        pitch = np.arctan2(camera_height, dis)
        y_down = (pitch + np.pi / 2) / np.pi * height
        points.append([x, y_down])
    points = np.array(points)
    i_first = directions.index(min(directions))
    points = np.roll(points, -i_first * 2, axis=0)
    points_unique = np.unique(points, axis=0)
    if len(points_unique) < len(points):
        print("duplicate points")
        return
    xs = points[::2, 0].astype(np.int)
    xs_unique = np.unique(xs)
    if len(xs_unique) < len(xs):
        print("duplicate x")
        return

    return points.astype(np.int32)


def cuboid_world_layout_from_room_layout(room_layout):
    # generate cuboid layout in the world frame from bounding box
    # plot_layout(r, camera)
    bdb2d = shapely.geometry.box(*room_layout['room'].bounds, ccw=True)
    # plot_layout(bdb2d, camera)
    room_layout = room_layout.copy()
    room_layout['room'] = bdb2d
    bbox3d = manhattan_world_layout_from_room_layout(room_layout)
    return bbox3d


def manhattan_world_layout_from_room_layout(room_layout):
    # generate manhattan layout in the world frame from bondary
    boundary = np.array(room_layout['room'].boundary.xy, dtype=np.float32)[:, :-1].T
    boundary_l = np.pad(boundary, [[0, 0], [0, 1]], 'constant', constant_values=0)
    boundary_h = np.pad(boundary, [[0, 0], [0, 1]], 'constant', constant_values=room_layout['height'])
    world_layout = np.concatenate([boundary_l, boundary_h], 0)
    return world_layout


def wall_bdb3d_from_manhattan_world_layout(layout, thickness=0.2):
    n_walls = len(layout) // 2
    walls = []
    for i_wall in range(n_walls):
        front_face = layout[(i_wall,
                             np.mod(i_wall + 1, n_walls),
                             i_wall + n_walls,
                             np.mod(i_wall + 1, n_walls) + n_walls), :]
        bdb3d_wall = bdb3d_from_front_face(front_face, thickness)
        walls.append({'bdb3d': bdb3d_wall})
    return walls


def wall_contour_from_manhattan_pix_layout(layout, transform: IGTransform):
    n_walls = len(layout) // 2
    walls = []
    for i_wall in range(n_walls):
        front_face = layout[(i_wall * 2 + 1,
                             i_wall * 2,
                             np.mod(i_wall * 2 + 2, len(layout)),
                             np.mod(i_wall * 2 + 3, len(layout))), :]

        # generate contour from front face
        contour = []
        front_face_3d = transform.campix23d(front_face, 1.)
        for p1, p2 in zip(front_face_3d, np.roll(front_face_3d, -1, axis=0)):
            contour.append(interpolate_line(p1, p2)[:-1])
        contour = np.concatenate(contour, 0)
        contour = transform.cam3d2pix(contour)

        # extend contour points divided by edge out of right edge
        outside = False
        contour_ext = []
        for p1, p2 in zip(contour, np.roll(contour, -1, axis=0)):
            if np.abs(p2[0] - p1[0]) > transform.camera['width'] / 2:
                outside = not outside
            if outside:
                p2[0] += transform.camera['width']
            contour_ext.append(p2)
        contour = np.stack(contour_ext)
        poly = Polygon(contour)
        # pyplot.plot(*poly.exterior.xy)
        # pyplot.axis('equal')
        # pyplot.show()

        walls.append({'contour': {
            'x': contour[..., 0].astype(np.int32),
            'y': contour[..., 1].astype(np.int32),
            'area': float(poly.area)
        }})
    return walls


def manhattan_world_layout_from_pix_layout(scene, camera_height=1.6):
    manhattan_pix = scene['layout']['manhattan_pix']
    corners_rad = scene.transform.campix2rad(manhattan_pix)
    corners_lon, corners_lat = corners_rad.T
    ceil_corners_lat, floor_corners_lat = corners_lat[::2], corners_lat[1::2]
    floor_corners_dis = camera_height / np.sin(floor_corners_lat)
    wall_corners_dis = camera_height / np.tan(floor_corners_lat)
    ceil_corners_dis = wall_corners_dis / np.cos(ceil_corners_lat)
    corners_rad = np.flip(corners_rad.reshape(-1, 2, 2).transpose(1, 0, 2), 0).reshape(-1, 2)
    corners_dis = np.concatenate([floor_corners_dis, ceil_corners_dis])
    world_layout = scene.transform.camrad2world(corners_rad, corners_dis)
    return world_layout


def manhattan_2d_from_manhattan_world_layout(layout):
    n_walls = len(layout) // 2
    layout_2d = layout[:n_walls, :2]
    return layout_2d


def manhattan_world_layout_info(layout):
    layout_z = layout[:, -1]
    layout_2d = manhattan_2d_from_manhattan_world_layout(layout)
    layout_poly = Polygon(layout_2d)
    floor = layout_z.min()
    ceil = layout_z.max()
    return {'ceil': ceil, 'floor': floor, 'layout_poly': layout_poly, 'layout_2d': layout_2d}


def layout_line_segment_indexes(N):
    layout_lines = [[i, (i + 1) % N] for i in range(N)] + \
                   [[i + N, (i + 1) % N + N] for i in range(N)] + \
                   [[i, i + N] for i in range(N)]
    return layout_lines
