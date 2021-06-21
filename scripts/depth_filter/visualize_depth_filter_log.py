#!/usr/bin/env python3

'''
Visualize VINS depth filter log
original: time1:point_id1,mu1,sigma1,inlier_ratio;point_id2,mu2,sigma2,inlier_ratio;...
          time2:point_id1,mu1,sigma1,inlier_ratio;point_id3,mu3,sigma3,inlier_ratio;...
'''

from functools import total_ordering
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.stats import norm
import os
import gc


def main():
    parser = argparse.ArgumentParser(
        description='Visualize VINS depth filter log')
    parser.add_argument('log_path', type=str, help='path to log file')
    parser.add_argument('--point_itr_thresh', type=int, default=200,
                        help='threshold of minimum number of iteration for a point to be visualized')
    parser.add_argument('--skip_frame', type=int, default=0,
                        help='skip x number of frames when visualize to reduce memory usage')
    parser.add_argument('--dump_file_path', type=str, default="",
                        help='foler to save all point plots and result')
    args = parser.parse_args()

    # parse file and create a dictionary of points
    # key: point_id, value: list of (mu, sigma)
    df_log = dict()
    f = open(args.log_path, 'r')
    for line in f:
        line_info = line.split(':')
        timestamp = line_info[0]
        points = line_info[1].split(';')
        # print("Timestamp {t} has {p} points".format(
        #     t=timestamp, p=len(points)))
        if (len(points) == 0):
            continue

        for point in points:
            if point == '\n':
                continue
            point_id = int(point.split(',')[0])
            mu = float(point.split(',')[1])
            sigma2 = float(point.split(',')[2])
            inlier_ratio = float(point.split(',')[3])
            if (point_id in df_log):
                df_log[point_id].append((mu, np.sqrt(sigma2), inlier_ratio))
            else:
                df_log[point_id] = [(mu, np.sqrt(sigma2), inlier_ratio)]

    f.close()

    # loop over and only check points with more than 10 iterations
    for point_id in df_log:
        if (len(df_log[point_id]) < args.point_itr_thresh):
            continue
        evos = df_log[point_id]
        print("id:", point_id, ", iterations:", len(evos))

        # visualize gaussian iterations
        point_iters = df_log[point_id]
        iter_num = 0
        total_iter_num = len(point_iters)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for point_iter in point_iters:
            if ((iter_num % args.skip_frame) != 0):
                iter_num += 1
                continue
            # get lowerbound and upperbound for visualization
            mu = point_iter[0]
            sigma = point_iter[1]
            lb = max(0, mu - 3*sigma)
            ub = mu + 3*sigma
            x_axis = np.arange(lb, ub, 0.00001)
            curve_color = (1-(iter_num/total_iter_num),
                           iter_num/total_iter_num, 0)

            ax.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=curve_color)
            iter_num += 1
        
        # put inlier ratio on fig
        ax.text(0.9, 0.9, str(point_iter[2]), ha='center', va='center', transform=ax.transAxes)

        print("inlier ratio:", point_iter[2])
        # show plot if not dumping to file
        if (args.dump_file_path == ""):
            fig.show()
        else:
            # save plot to file
            plt_path = os.path.join(
                args.dump_file_path, (str(point_id) + '.png'))
            fig.savefig(plt_path)
            print("saved fig to:", plt_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
