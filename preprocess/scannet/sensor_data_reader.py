import argparse
import os, sys
import glob
from SensorData import SensorData


def read(opt):
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    file_pattern = os.path.join(opt.input_folder, '*.sens')
    sens_files = glob.glob(file_pattern)
    if sens_files:
        input_data_path = sens_files[0]
        sys.stdout.write('loading %s...' % input_data_path)
    else:
        print("No .sens file found.")
        return
    
    sd = SensorData(input_data_path)
    sys.stdout.write('loaded!\n')
    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(opt.output_folder, 'depth'))
    if opt.export_color_images:
        sd.export_color_images(os.path.join(opt.output_folder, 'color'))
    if opt.export_poses:
        sd.export_poses(os.path.join(opt.output_folder, 'pose'))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(opt.output_folder, 'intrinsic'))


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('--input_folder', required=True, help='path to the folder contains .sens file')
    parser.add_argument('--output_folder', required=True, help='path to output folder')
    # default settings
    parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
    parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
    parser.add_argument('--export_poses', dest='export_poses', action='store_true')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
    parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

    opt = parser.parse_args()
    read(opt)
