'''
Main file for visualizing the ReplicaPano dataset
author: YuanDong
'''
import os
import cv2
import argparse

from utils.igibson_utils import ReplicaPanoScene

if __name__ == '__main__':

    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Visualization ReplicaPano Dataset')
    # Adding argument for the path to the data in .pkl format, which is required
    parser.add_argument('--pkl_path', type=str, required=True, help='Please specify the data pkl path')
    # Adding optional argument for specifying the output file path for PLY format
    parser.add_argument('--output_ply_filepath', type=str, default=None, help='Output file path for PLY format')
    # Output file path for layout mesh in OBJ format
    parser.add_argument('--layout_output_obj_filepath', type=str, default=None, help='Output file path for layout mesh in OBJ format')
    # Flag to convert 3d objects to world space
    parser.add_argument('--to_world_space', type=bool, default=False, help='Flag to convert point cloud to world space')
    # Output folder
    parser.add_argument('--output_dir', type=str, default='example_output', help='Output folder')
    args = parser.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)

    # Load the scene from the specified pickle file
    scene = ReplicaPanoScene.from_pickle(args.pkl_path)

    # Load the image from the scene
    image = scene.image()

    # Process the image to include 3D objects
    image_objs3d = scene.objs3d(image.copy())
    cv2.imwrite(os.path.join(args.output_dir,'example_objs3d.png'), image_objs3d[:, :, (2, 1, 0)])

    # Process the image to add 2D bounding boxes
    image_bdb2d = scene.bdb2d(image.copy())
    cv2.imwrite(os.path.join(args.output_dir,'example_bdb2d.png'), image_bdb2d[:, :, (2, 1, 0)])

    # Process the image to include layout information
    image_layout = scene.layout(image.copy())
    cv2.imwrite(os.path.join(args.output_dir,'example_layout.png'), image_layout[:, :, (2, 1, 0)])

    # Process the image to include the bird's eye view field of view
    image_bfov = scene.bfov(image.copy())
    cv2.imwrite(os.path.join(args.output_dir,'example_bfov.png'), image_bfov[:, :, (2, 1, 0)])

    # Generate point cloud data from the scene
    pointcloud = scene.pointcloud(output_ply_filepath=args.output_ply_filepath, to_world_space=args.to_world_space)

    # Save the layout mesh to an OBJ file at the specified output path
    layout_mesh = scene.save_layout_mesh(output_obj_filepath=args.layout_output_obj_filepath, to_world_space=args.to_world_space)

