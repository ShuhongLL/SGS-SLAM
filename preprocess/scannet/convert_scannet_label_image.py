# Example script to convert label images from the *_2d-label.zip or *_2d-label-filt.zip data for each scan.
# Note: already preprocessed data for a set of frames subsampled from the full datasets is available to download through the ScanNet download.
# Input:
#   - path to label image to convert
#   - label mapping file (scannetv2-labels.combined.tsv)
#   - output image file
# Outputs the label image with nyu40 labels as an u8-bit image 

import math
import os, sys, argparse, glob
import csv
import inspect
import numpy as np
import imageio
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

simplified_nyu40 = {
    "wall": [174, 199, 232],
    "floor": [152, 223, 138],
    "cabinet": [31, 119, 180],
    "bed": [255, 187, 120], # <-
    "chair": [188, 189, 34],
    "sofa": [140, 86, 75], # <-
    "table": [255, 152, 150],
    "door": [214, 39, 40],
    "window": [197, 176, 213],
    "bookshelf": [148, 103, 189], # <-
    "picture": [196, 156, 148],
    "counter": [23, 190, 207], # <-
    "desk": [247, 182, 210], # <-
    "curtain": [219, 219, 141],
    "refrigerator": [255, 127, 14],
    "showercurtain": [158, 218, 229],
    "toilet": [44, 160, 44],
    "sink": [112, 128, 144],
    "bathtub": [227, 119, 194],
    # "otherfurniture": [82, 84, 163],
    "guitar": [255, 0, 0],
    "bicycle": [0, 255, 0],
    "basket": [255, 255, 0], # <-
    "whiteboard": [255, 0, 255],
    "backpack": [0, 255, 255], # <-
    "microwave": [128, 0, 128],
    "bag": [255, 165, 0],
    "stool": [],
    "pillow": []
}


nyu40_class2rgb = {
    "unlabeled": [0, 0, 0],
    "wall": [174, 199, 232],
    "floor": [152, 223, 138],
    "cabinet": [31, 119, 180],
    "bed": [255, 187, 120],
    "chair": [188, 189, 34],
    "sofa": [140, 86, 75],
    "table": [255, 152, 150],
    "door": [214, 39, 40],
    "window": [197, 176, 213],
    "bookshelf": [148, 103, 189],
    "picture": [196, 156, 148],
    "counter": [23, 190, 207],
    "desk": [247, 182, 210],
    "curtain": [219, 219, 141],
    "refrigerator": [255, 127, 14],
    "showercurtain": [158, 218, 229],
    "toilet": [44, 160, 44],
    "sink": [112, 128, 144],
    "bathtub": [227, 119, 194],
    "otherfurniture": [82, 84, 163],
    "guitar": [255, 0, 0],
    "bicycle": [0, 255, 0],
    "basket": [255, 255, 0],
    "whiteboard": [255, 0, 255],
    "backpack": [0, 255, 255],
    "microwave": [128, 0, 128],
    "bag": [255, 165, 0],
}

GUITAR_ID = 112
BICYCLE_ID = 121
BASKET_ID = 106
WHITE_BOARD_ID = 52
BACKPACK_ID = 48
MICROWAVE_ID = 59
BAG_ID = 47


# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='id', label_to='nyu40class'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[int(row[label_from])] = row[label_to]
    return mapping


def map_label_image(image, label_mapping):
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for k, v in label_mapping.items():
       class_name = v if v in nyu40_class2rgb.keys() else "unlabeled"
       color_image[image == k] = nyu40_class2rgb[class_name]
    return color_image.astype(np.uint8)


def convert(opt):
    image_file_paths = glob.glob(os.path.join(opt.input_label_folder, '*.png'))
    
    if not os.path.exists(opt.output_label_folder):
        os.makedirs(opt.output_label_folder)

    for image_file_path in tqdm(image_file_paths):
        image = np.array(imageio.imread(image_file_path))
        label_map = read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40class')
        # add certain objects
        label_map[GUITAR_ID] = "guitar"
        label_map[BICYCLE_ID] = "bicycle"
        label_map[BASKET_ID] = "basket"
        label_map[WHITE_BOARD_ID] = "whiteboard"
        label_map[BACKPACK_ID] = "backpack"
        label_map[MICROWAVE_ID] = "microwave"
        label_map[BAG_ID] = "bag"
        mapped_image = map_label_image(image, label_map)

        # Construct the output file path and save the color image
        output_file_path = os.path.join(opt.output_label_folder, os.path.basename(image_file_path))
        imageio.imwrite(output_file_path, mapped_image)
    # uncomment to save out visualization
    # util.visualize_label_image(os.path.splitext(opt.output_file)[0] + '_vis.jpg', mapped_image)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_label_map_path = os.path.join(script_dir, 'scannetv2-labels.combined.tsv')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_label_folder', required=True, help='path to input filtered label folder')
    parser.add_argument('--label_map_file', default=default_label_map_path, help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--output_label_folder', required=True, help='output image file folder')
    opt = parser.parse_args()
    convert(opt)
