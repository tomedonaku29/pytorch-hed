#!/usr/bin/env python

import torch

import argparse
import math
import numpy
import os
import PIL
import PIL.Image
import sys

from network import estimate

##########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file')
parser.add_argument('--output', type=str, help='output file')

##########################################################

if __name__ == '__main__':
    args = parser.parse_args()
    img = PIL.Image.open(args.input)
    rgb_img = PIL.Image.new("RGB", img.size)
    rgb_img.paste(img)
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(rgb_img)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenOutput = estimate(tenInput)
    PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(args.output)
# end
