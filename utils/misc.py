# This file is copied from the 'DeepPanoContext'
# The original source can be found at: https://github.com/chengzhag/DeepPanoContext
# Modifications may be needed to adapt it to the current project requirements.

import os
import cv2
import pickle
import pickle5
import numpy as np
import seaborn as sns
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

igibson_colorbox_temp = np.array(sns.hls_palette(n_colors=60, l=.45, s=.8))
igibson_colorbox = []

color_map = {0:31,1:33,2:46,3:4,4:47,5:17,6:7,7:38,8:0,9:40,
             10:39,11:54,12:6,13:51,14:21,15:56,16:24,17:50,18:20,19:2,20:46,21:48,
             22:28,23:44,24:53}

for idx in range(len(color_map)):
    igibson_colorbox.append(igibson_colorbox_temp[color_map[idx]])
igibson_colorbox = np.array(igibson_colorbox)

def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi

def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi

def layout_line_segment_indexes(N):
    layout_lines = [[i, (i + 1) % N] for i in range(N)] + \
                   [[i + N, (i + 1) % N + N] for i in range(N)] + \
                   [[i, i + N] for i in range(N)]
    return layout_lines


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])

def read_pkl(pkl_file, protocol=4):
    with open(pkl_file, 'rb') as file:
        if protocol == 4:
            data = pickle.load(file)
        elif protocol == 5:
            data = pickle5.load(file)
        else:
            raise NotImplementedError()
    return data

def visualize_igibson_detectron_gt(sample, image=None, dataset=None):
    if dataset is None:
        metadata_replica = read_pkl('./assert/metadata_replica.pkl')
        dataset = metadata_replica
    if image is None:
        image = np.array(Image.open(sample["file_name"]))
    visualizer = Visualizer(image, metadata=dataset)
    image = visualizer.draw_dataset_dict(sample).get_image()
    return image

def visualize_image(image, key=None):
    if isinstance(image, dict):
        visual = {}
        for k, v in image.items():
            visual[k] = visualize_image(v, k)
        return visual
    if key == 'depth':
        image = image.astype(np.float)
        image = (image / image.max() * 255).astype(np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

class Camera(dict):
    _shared_params = {'width', 'height', 'K', 'vertical_fov'}

    def __init__(self, seq=None):
        if isinstance(seq, dict):
            self.cameras = [self]
            super(Camera, self).__init__(seq)
        elif isinstance(seq, list):
            self.cameras = seq
            shared_params = {k: v for k, v in seq[0].items() if k in self._shared_params}
            for c in seq[1:]:
                for k, v in shared_params.items():
                    assert np.all(c[k] == v)
            super(Camera, self).__init__(shared_params)

    def __getitem__(self, item):
        value = super(Camera, self).get(item)
        return value

    def __iter__(self):
        return self.cameras.__iter__()

def detectron_gt_sample(data, idx=None):
    record = {
        "file_name": data['image_path']['rgb'],
        "image_id": idx,
        "height": data['camera']['height'],
        "width": data['camera']['width']
    }
    annotations = []
    for obj in data['objs']:
        bdb2d, contour = obj['bdb2d'], obj['contour']
        poly = [(x + 0.5, y + 0.5) for x, y in zip(contour['x'], contour['y'])]
        poly = [p for x in poly for p in x]
        obj = {
            "bbox": [bdb2d['x1'], bdb2d['y1'], bdb2d['x2'], bdb2d['y2']],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": obj['label'],
        }
        annotations.append(obj)
    record["annotations"] = annotations
    return record

def pickle_path(path):
    if not os.path.splitext(path)[1]:
        camera_folder = path
        pickle_file = os.path.join(path, 'data.pkl')
    elif path.endswith('pkl'):
        camera_folder = os.path.dirname(path)
        pickle_file = path
    else:
        raise Exception('Input path can be either folder or pkl file')
    os.makedirs(camera_folder, exist_ok=True)
    return camera_folder, pickle_file

def wrapped_line(image, p1, p2, colour, thickness, lineType=cv2.LINE_AA):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0]+image.shape[1], p1[1]])
    p2b = np.array([p2[0]-image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness, lineType=lineType)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness, lineType=lineType)
        cv2.line(image, tuple(p1b), p2, colour, thickness, lineType=lineType)
