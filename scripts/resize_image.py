#!/usr/bin/env python3

# Resize image
# Henry Zhang <hzhang0407@gmail.com>

import cv2
import os
import argparse
import yaml
import numpy as np

def main():
  parser = argparse.ArgumentParser(description='resize image')
  parser.add_argument('image_path', type=str, help='path to image')
  parser.add_argument('scale_width', type=float,
                      help='scale width')
  parser.add_argument('scale_height', type=float,
                      help='scale height')
  parser.add_argument('--target_width', type=int, default=0,
                      help='target image width')
  parser.add_argument('--scale_height', type=int, default=0,
                      help='target image height')

  args = parser.parse_args()

  output_path = args.image_path.split('.')[0] + '_resized.' + args.image_path.split('.')[1]
  print("output image to " + output_path)

  # read image
  img = cv2.imread(args.image_path)

  if (args.target_width is 0):
      target_width = int(img.shape[1] * args.scale_width)
      target_height = int(img.shape[0] * args.scale_height)
  else:
      target_width = args.target_width
      target_height = args.target_height

  img_out = cv2.resize(img, (target_width, target_height))
  cv2.imwrite(output_path, img_out)
  
if __name__ == "__main__":
    main()
