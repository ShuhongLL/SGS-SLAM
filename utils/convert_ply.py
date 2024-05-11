import numpy as np
import argparse
from plyfile import PlyElement, PlyData # Requires plyfile==0.8.1

C0 = 0.28209479177387814


def construct_list_of_attributes(f_dc, scale, rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l


def convert(params, dest, is_semantic=False):
    xyz = params['means3D']
    normals = np.zeros_like(xyz)
    if not is_semantic:
        f_dc =  (params['rgb_colors'] - 0.5) / C0
    else:
        f_dc =  (params['semantic_colors'] - 0.5) / C0
    f_rest = np.zeros_like(f_dc)
    opacities = params['logit_opacities']
    scale = params['log_scales'].repeat(3, axis=-1)
    rotation = params['unnorm_rotations']

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(f_dc, scale, rotation)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to experiment npz file")
    parser.add_argument("--dest", type=str, required=True, help="Path to output ply file")
    args = parser.parse_args()
    params = np.load(args.src)
    convert(params, args.dest)