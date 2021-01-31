#!/usr/bin/env python

import torch

import argparse
import glob
import pathlib
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from network import estimate

##########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help='input directory')
parser.add_argument('--output_dir', type=str, help='output directory')

##########################################################

if __name__ == '__main__':
    args = parser.parse_args()

    for ext in ('jpg', 'jpeg', 'png'):
        files = glob.glob(args.input_dir + '/*.' + ext)
        for f in files:
            p = pathlib.Path(f)
            print("Processing {}...".format(p))
            img = PIL.Image.open(f)
            rgb_img = PIL.Image.new("RGB", img.size)
            rgb_img.paste(img)
            tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(rgb_img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            tenOutput = estimate(tenInput)
            PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save('{}/{}_edge.jpg'.format(args.output_dir, p.stem))
# end
