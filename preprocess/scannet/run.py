import argparse
import os, sys
import glob
import zipfile

from convert_scannet_label_image import convert
from sensor_data_reader import read


parser = argparse.ArgumentParser()
script_dir = os.path.dirname(os.path.realpath(__file__))
default_label_map_path = os.path.join(script_dir, 'scannetv2-labels.combined.tsv')
# data paths
parser.add_argument('--input_folder', required=True, help='path to the folder contains .sens file')
parser.add_argument('--output_folder', required=True, help='path to output folder')
# default settings
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.add_argument('--export_seg', dest='export_seg', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)
parser.add_argument('--label_map_file', default=default_label_map_path, help='path to scannetv2-labels.combined.tsv')

opt = parser.parse_args()


def run(opt):
    # Read .sens file
    read(opt)

    if opt.export_seg:
        # Unzip filtered label file
        zip_files = glob.glob(os.path.join(opt.input_folder, 'scene*_00_2d-label-filt.zip'))

        if len(zip_files) > 0:
            print(f"Unzip: {zip_files[0]}")
        else:
            print("No label-filt.zip file found.")
            return

        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall(opt.output_folder)
            
        extracted_folder_path = os.path.join(opt.output_folder, 'label-filt')
        new_folder_path = os.path.join(opt.output_folder, 'semantic_id')
        os.rename(extracted_folder_path, new_folder_path)

        # fetch semantic label and image
        opt.input_label_folder = os.path.join(opt.output_folder, 'semantic_id')
        opt.output_label_folder = os.path.join(opt.output_folder, 'semantic_color')
        convert(opt)
    

if __name__ == '__main__':
    run(opt)
