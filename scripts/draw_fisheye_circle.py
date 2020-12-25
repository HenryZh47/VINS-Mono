#!/usr/bin/env python3

# Draw fisheye mask based on calibration result
# Henry Zhang <hzhang0407@gmail.com>

# Read in Kalibr camera calibration yaml result

import cv2
import os
import argparse
import yaml
import numpy as np

def main():
  parser = argparse.ArgumentParser(description='Draw the fisheye mask from Kalibr calibration output')
  parser.add_argument('yaml_path', type=str,
                      help='Kalibr yaml file path for the fisheye mask')
  parser.add_argument('--output_folder', default='./', type=str,
                      help='folder to output the fisheye mask')

  args = parser.parse_args()

  # read the yaml file
  with open(args.yaml_path, 'r') as file:
    cams = yaml.full_load(file)
    for cam_id in cams:
      # get image res and prinicple points
      cam = cams[cam_id]
      res = cam['resolution']
      intrinsics = cam['intrinsics']
      u0 = int(intrinsics[2])
      v0 = int(intrinsics[3])

      # draw the circle
      image = np.zeros((res[1], res[0]), np.uint8)
      cv2.circle(image, (u0, v0), int(max(res)/2), (255), -1)

      cv2.imshow('mask', image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      # save image
      img_name = 'fisheye_mask_%s.png' %(cam_id)
      img_name = os.path.join(args.output_folder, img_name)
      cv2.imwrite(img_name, image)


if __name__ == "__main__":
  main()