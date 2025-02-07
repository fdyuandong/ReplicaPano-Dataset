
# This file is modified from the 'DeepPanoContext'
# The original source can be found at: https://github.com/chengzhag/DeepPanoContext

from glob import glob

from utils.image_utils import ImageIO
from utils.image_utils import load_image, save_image
from utils.transform_utils import IGTransform, interpolate_line, bdb3d_corners
from utils.misc import *


class ReplicaPanoScene(object):

    '''
    A class used to store, process and visualize replicepano scene contents.
    '''

    ignore_when_saving = ['image_path', 'mesh_path', 'image_tensor', 'image_np']
    image_types = ['rgb', 'seg', 'sem', 'depth']
    consider_when_transform = {
        'objs': {'bdb3d'},
        'layout': {'manhattan_world', 'cuboid_world'},
        'walls': {'bdb3d'}
    }
    basic_ignore_from_batch = {
        'objs': {'split', 'mesh_extractor', 'g_feature',
                 'lien_afeature', 'lien_activation', 'ben_afeature', 'ben_rfeature', 'ben_arfeature'},
        'walls': {'split'}
    }
    further_ignore_from_batch = {
        'objs': {'cls_code', 'rgb', 'seg'}
    }

    def __init__(self, data):
        assert isinstance(data, dict)
        self.image_io = ImageIO.from_file(data.get('image_path', {}))
        self.pkl_path = None
        self.data = data
        if 'image_np' in data:
            self.image_io = ImageIO(data['image_np'])
        self.transform = IGTransform(self.data)
        self.data['camera'] = Camera(self.data['camera'])
        # update data camera
        self.transform_3d = IGTransform(self.data)
        self.transform_3d.camera['cam3d2world'] = self.data['T_w_c']
        self.transform_3d.camera['world2cam3d'] = np.linalg.inv(self.data['T_w_c'])

    @classmethod
    def from_pickle(cls, path: str, igibson_obj_dataset=None):
        camera_folder, pickle_file = pickle_path(path)
        data = read_pkl(pickle_file)
        image_path = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(camera_folder, '*.png'))}
        data['image_path'] = {k: v for k, v in image_path.items() if k in cls.image_types}
        if igibson_obj_dataset is not None:
            mesh_path = [os.path.join(igibson_obj_dataset, o['model_path']+'.obj')
                         for o in data['objs'] if 'model_path' in o]
            if mesh_path:
                data['mesh_path'] = mesh_path
        scene = cls(data)
        scene.pkl_path = pickle_file
        return scene

    @classmethod
    def from_image(cls, path):
        h, w = 512, 1024
        rgb = load_image(path)
        if any(a != b for a, b in zip(rgb.shape[:2], [h, w])):
            image = Image.fromarray(rgb)
            image = image.resize((w, h),Image.ANTIALIAS)
            path = os.path.splitext(path)[0] + '.png'
            save_image(np.array(image), path)
        name = os.path.splitext(os.path.basename(path))[0]
        data = {
            'name': name,
            'scene': '',
            'camera': {'height': h, 'width': w},
            'image_path': {'rgb': path}
        }
        scene = cls(data)
        scene.transform.set_camera_to_world_center()
        return scene

    def empty(self):
        return all(k in ('name', 'scene', 'camera', 'image_path') for k in self.data.keys())

    def fov_split(self, fov, gt_offset=0, offset_bdb2d=False):
        fov = np.deg2rad(fov)
        assert np.isclose(np.mod(np.pi * 2, fov), 0)

        # split cameras
        yaws = np.arange(0, np.pi * 2, fov).astype(np.float32)
        n_split = len(yaws)
        split_width = self['camera']['width'] / n_split
        targets_rad = np.stack([yaws, np.zeros_like(yaws)], -1)
        targets = self.transform.camrad2world(targets_rad, 1)
        trans_cams = []
        for target in targets:
            trans = self.transform.copy()
            trans.look_at(target)
            trans_cams.append(trans)

        # split objects by cameras
        obj_splits = [[] for _ in range(len(trans_cams))]
        for obj in self['objs']:
            obj = obj.copy()

            # offset ground truth mapping
            if 'gt' in obj:
                obj['gt'] = obj['gt'] + gt_offset

            # find out which camera the object is in
            bfov_center_rad = np.array([obj['bfov']['lon'], obj['bfov']['lat']])
            bfov_center_world = self.transform.camrad2world(bfov_center_rad, 1)
            for i_scene, trans in enumerate(trans_cams):
                # transform object center from world frame to camera frame
                cam_center_rad = trans.world2camrad(bfov_center_world)
                if -fov / 2 < cam_center_rad[0] <= fov / 2:
                    # offset bdb2d to target camera
                    if offset_bdb2d:
                        x1, x2 = obj['bdb2d']['x1'], obj['bdb2d']['x2']
                        bdb2d_width = x2 - x1
                        bdb2d_center = (x1 + x2) / 2
                        bdb2d_center_in_cam = np.mod(
                            bdb2d_center - (0 if np.mod(n_split, 2) else split_width / 2),
                            split_width
                        )
                        x1 = int(bdb2d_center_in_cam - bdb2d_width / 2)
                        x2 = int(bdb2d_center_in_cam + bdb2d_width / 2)
                        obj['bdb2d']['x1'], obj['bdb2d']['x2'] = x1, x2
                    obj_splits[i_scene].append(obj)
                    break

        # create scenes with new objects and cameras
        scenes = []
        for trans, objs in zip(trans_cams, obj_splits):
            data = self.data.copy()
            data['camera'] = trans.camera
            data['objs'] = objs
            scene = ReplicaPanoScene(data)
            scenes.append(scene)

        return scenes

    def set_camera_to_world_center(self):
        # transform to camera centered and orientated world frame
        def apply_on_specified(dic, keys, func):
            if isinstance(dic, list):
                for i in dic:
                    apply_on_specified(i, keys, func)
            elif isinstance(keys, dict):
                for k, v in keys.items():
                    if v is True and k in dic:
                        dic[k] = func(dic[k])
                    elif k in dic:
                        apply_on_specified(dic[k], v, func)
            else:
                for k in keys:
                    if k in dic:
                        if isinstance(dic[k], dict):
                            dic[k].update(func(dic[k]))
                        else:
                            dic[k] = func(dic[k])
        apply_on_specified(self.data, self.consider_when_transform, self.transform.world2cam3d)
        self.transform.set_camera_to_world_center()
        apply_on_specified(self.data, self.consider_when_transform, self.transform.cam3d2world)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_horizon(self, path):
        output_name =  f"{self.data['scene']}_{self.data['name']}"

        # save layout as LayoutNet/HorizonNet format
        layout_folder = os.path.join(path, 'label_cor')
        os.makedirs(layout_folder, exist_ok=True)
        layout_txt = os.path.join(layout_folder, output_name + '.txt')
        np.savetxt(layout_txt, self.data['layout']['manhattan_pix'], '%d')

        # link rgb to horizonnet folder
        image_folder = os.path.join(path, 'img')
        os.makedirs(image_folder, exist_ok=True)
        dst = os.path.join(image_folder, output_name + '.png')
        if os.path.exists(dst):
            os.remove(dst)
        os.link(self.image_io['image_path']['rgb'], dst)

    def image(self, key='rgb'):
        image = self.image_io[key]
        image = visualize_image(image, key)
        return image

    def depth(self):
        depthmap = self.image_io['depth']
        return depthmap


    def pointcloud(self, output_ply_filepath = None, to_world_space = False):

        def get_unit_map():
            h = 512
            w = 1024
            Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
            Theta = np.repeat(Theta, w, axis=1)
            Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
            Phi = np.repeat(Phi, h, axis=0)

            X = np.expand_dims(np.sin(Theta) * np.sin(Phi), 2)
            Y = np.expand_dims(np.cos(Theta), 2)
            Z = np.expand_dims(np.sin(Theta) * np.cos(Phi), 2)
            unit_map = np.concatenate([X, Z, Y], axis=2)

            return unit_map

        depthmap = self.depth()
        unit_map = get_unit_map()
        point_cloud_map = np.repeat(np.expand_dims(depthmap, axis=2), 3, axis=2) * unit_map
        point_cloud = point_cloud_map.reshape(-1,3)

        if(output_ply_filepath is not None):
            if(to_world_space):
                point_cloud = self.transform_3d.cam3d2world(point_cloud)
            self.write_ply(point_cloud,output_ply_filepath)

        return point_cloud

    def save_layout_mesh(self, output_obj_filepath = None, to_world_space = False):
        from shapely.geometry import Polygon
        import trimesh
        layout_cor_num = self.data['layout']['manhattan_world'].shape[0]//2
        floor_xy = self.data['layout']['manhattan_world'][:layout_cor_num,:2]

        # update floor_xy
        for i in range(floor_xy.shape[0]):
            i_cor = floor_xy[i,:]
            j_cor = floor_xy[(i+1)%floor_xy.shape[0],:]
            if(abs(i_cor[0]-j_cor[0])<abs(i_cor[1]-j_cor[1])):
                update_idx = 0
            else:
                update_idx = 1
            temp = (floor_xy[i,update_idx]+floor_xy[(i+1)%floor_xy.shape[0],update_idx])/2
            floor_xy[i,update_idx],floor_xy[(i+1)%floor_xy.shape[0],update_idx] = temp, temp

        polygon = Polygon(floor_xy.tolist())
        transformation = np.eye(4)
        transformation[0, 0] = 0
        transformation[1, 1] = 0
        transformation[0, 1] = -1
        transformation[1, 0] = 1

        transformation[2, 3] = np.min(self.data['layout']['manhattan_world'][:,2])
        mesh = trimesh.creation.extrude_polygon(polygon, height=np.max(self.data['layout']['manhattan_world'][:,2]) - np.min(self.data['layout']['manhattan_world'][:,2]),
                                                transform=transformation)
        if(to_world_space):
            mesh.vertices = self.transform_3d.cam3d2world(np.array(mesh.vertices))

        if(output_obj_filepath is not None):
            mesh.export(output_obj_filepath)

        return mesh

    def objs3d(self, image, bbox3d=True, axes=True, centroid=True, info=False, thickness=2):
        if 'objs' not in self.data or not self.data['objs'] or 'bdb3d' not in self.data['objs'][0]:
            return image
        image = image.copy()
        objs = self.data['objs']
        dis = [np.linalg.norm(self.transform.world2cam3d(o['bdb3d']['centroid'])) for o in objs]
        i_objs = sorted(range(len(dis)), key=lambda k: dis[k])
        for i_obj in reversed(i_objs):
            obj = objs[i_obj]
            color = (igibson_colorbox[obj['label']] * 255).astype(np.uint8).tolist()
            bdb3d = obj['bdb3d']
            if axes:
                self._objaxes(image, bdb3d, thickness=thickness)
            if centroid:
                self._centroid(image, bdb3d['centroid'], color, thickness=thickness)
            if bbox3d:
                self._bdb3d(image, bdb3d, color, thickness=thickness)
            if info:
                self._objinfo(image, bdb3d, color, obj['instance_name'])
        return image

    def bdb2d(self, image, dataset=None):
        if 'objs' not in self.data or not self.data['objs'] or 'bdb2d' not in self.data['objs'][0]:
            return image
        sample = detectron_gt_sample(self.data)
        image = visualize_igibson_detectron_gt(sample, image, dataset)
        return image

    def layout(self, image, color=(255, 255, 0), thickness=2):

        H, W = image.shape[:2]

        if 'layout' not in self.data or 'manhattan_pix' not in self.data['layout']:
            return image
        cor_id = np.array(self.data['layout']['manhattan_pix'], np.float32)
        image = image.copy()

        N = len(cor_id) // 2
        floor_z = -1.6
        floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
        c = np.sqrt((floor_xy ** 2).sum(1))
        v = np_coory2v(cor_id[0::2, 1], H)
        ceil_z = (c * np.tan(v)).mean()

        assert N == len(floor_xy)
        layout_points = [[x, -floor_z, -y] for x, y in floor_xy] + \
                    [[x, -ceil_z, -y] for x, y in floor_xy]
        frame = 'cam3d'

        layout_lines = layout_line_segment_indexes(N)

        layout_points = np.array(layout_points)
        for point1, point2 in layout_lines:
            point1 = layout_points[point1]
            point2 = layout_points[point2]
            self._line3d(image, point1, point2, color, thickness, frame=frame)

        return image

    def bfov(self, image, thickness=2, include=('objs', )):
        for key in include:
            if key not in self.data or not self.data[key] or 'bfov' not in self.data[key][0]:
                continue
            image = image.copy()
            objs = self.data[key]
            for obj in objs:
                color = (igibson_colorbox[obj['label']] * 255).astype(np.uint8).tolist() \
                    if 'label' in obj else (255, 255, 0)
                bfov = obj['bfov']
                self._bfov(image, bfov, color, thickness)
        return image

    def _bfov(self, image, bfov, color, thickness=2):
        target = self.transform.camrad2world(np.stack([bfov['lon'], bfov['lat']]), 1)
        pers_trans = self.transform.copy()
        pers_trans.look_at(target)

        # coordinate of right down corner in perspective camera frame
        half_x_fov = bfov['x_fov'] / 2
        half_y_fov = bfov['y_fov'] / 2
        half_height = np.tan(half_y_fov)
        dis_right = 1 / np.cos(half_x_fov)
        right_down_y = half_height / dis_right
        right_down_x = np.sin(half_x_fov)
        right_down_z = np.cos(half_x_fov)

        corners = np.array([
            [right_down_x, right_down_y, right_down_z],
            [-right_down_x, right_down_y, right_down_z],
            [-right_down_x, -right_down_y, right_down_z],
            [right_down_x, -right_down_y, right_down_z]
        ])

        corners = pers_trans.cam3d2world(corners)
        for start, end in zip(corners, np.roll(corners, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, frame='world')

    def _objaxes(self, image, bdb3d, thickness=2):
        origin = np.zeros(3, dtype=np.float32)
        centroid = self.transform.obj2frame(origin, bdb3d)
        for axis in np.eye(3, dtype=np.float32):
            endpoint = self.transform.obj2frame(axis / 2, bdb3d)
            color = axis * 255
            self._line3d(image, centroid, endpoint, color, thickness, frame='world')

    def _centroid(self, image, centroid, color, thickness=2):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        center = self.transform.world2campix(centroid)
        cv2.circle(image, tuple(center.astype(np.int32).tolist()), 5, color, thickness=thickness, lineType=cv2.LINE_AA)

    def _bdb3d(self, image, bdb3d, color, thickness=2):
        corners = self.transform.world2cam3d(bdb3d_corners(bdb3d))
        corners_box = corners.reshape(2, 2, 2, 3)
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    self._line3d(image, corners_box[idx1], corners_box[idx2], color, thickness=thickness, frame='cam3d')
        for idx1, idx2 in [(1, 7), (3, 5)]:
            self._line3d(image, corners[idx1], corners[idx2], color, thickness=thickness, frame='cam3d')

    def _contour(self, image, contour, color, thickness=1):
        contour_pix = np.stack([contour['x'], contour['y']], -1)
        contour_3d = self.transform.campix23d(contour_pix, 1)
        for start, end in zip(contour_3d, np.roll(contour_3d, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, quality=2, frame='cam3d')

    def _bdb2d(self, image, bdb2d, color, thickness=1):
        x1, x2, y1, y2 = bdb2d['x1'], bdb2d['x2'], bdb2d['y1'], bdb2d['y2']
        corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        corners_pix = np.array(corners)
        corners_3d = self.transform.campix23d(corners_pix, 1)
        for start, end in zip(corners_3d, np.roll(corners_3d, 1, axis=0)):
            self._line3d(image, start, end, color, thickness=thickness, frame='cam3d')

    def _objinfo(self, image, bdb3d, color, name):
        color = [255 - c for c in color]
        bdb3d_pix = self.transform.world2campix(bdb3d)
        bdb3d_info = [
                      f"center: {bdb3d_pix['center'][0]:.0f}, {bdb3d_pix['center'][1]:.0f}",
                      f"dis: {bdb3d_pix['dis']:.1f}",
                      f"ori: {np.rad2deg(bdb3d_pix['ori']):.0f}",
                      f"{name}"]
        bottom_left = bdb3d_pix['center'].copy().astype(np.int32)
        for info in reversed(bdb3d_info):
            bottom_left[1] -= 16
            cv2.putText(image, info, tuple(bottom_left.tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _line3d(self, image, p1, p2, color, thickness, quality=30, frame='world'):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        if frame == 'world':
            p1 = self.transform.world2cam3d(p1)
            p2 = self.transform.world2cam3d(p2)
        elif frame != 'cam3d':
            raise NotImplementedError
        points = interpolate_line(p1, p2, quality)
        pix = np.round(self.transform.cam3d2pix(points)).astype(np.int32)
        for t in range(quality - 1):
            p1, p2 = pix[t], pix[t + 1]
            if 'K' in self.data['camera']:
                if self.transform.in_cam(points[t], frame='cam3d') \
                        or self.transform.in_cam(points[t + 1], frame='cam3d'):
                    cv2.line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)
            else:
                wrapped_line(image, tuple(p1), tuple(p2), color, thickness, lineType=cv2.LINE_AA)

    def write_ply(self, points, filename, text=True):
        from plyfile import PlyData, PlyElement
        """ input: Nx3, write points to filename as PLY format. """
        points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        PlyData([el], text=text).write(filename)