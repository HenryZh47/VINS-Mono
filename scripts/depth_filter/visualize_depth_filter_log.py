#!/usr/bin/env python3

'''
Visualize VINS depth filter log
original: time1:point_id1,mu1,sigma1;point_id2,mu2,sigma2;...
          time2:point_id1,mu1,sigma1;point_id3,mu3,sigma3;...
'''

from functools import total_ordering
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.stats import norm


def main():
    parser = argparse.ArgumentParser(
        description='Visualize VINS depth filter log')
    parser.add_argument('log_path', type=str, help='path to log file')
    parser.add_argument('point_itr_thresh', type=int,
                        help='threshold of minimum number of iteration for a point to be visualized')
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

        for point in points:
            if point == '\n':
                continue
            point_id = int(point.split(',')[0])
            mu = float(point.split(',')[1])
            sigma2 = float(point.split(',')[2])
            if (point_id in df_log):
                df_log[point_id].append((mu, np.sqrt(sigma2)))
            else:
                df_log[point_id] = [(mu, np.sqrt(sigma2))]

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
        for point_iter in point_iters:
            # get lowerbound and upperbound for visualization
            mu = point_iter[0]
            sigma = point_iter[1]
            lb = max(0, mu - 3*sigma)
            ub = mu + 3*sigma
            x_axis = np.arange(lb, ub, 0.00001)
            curve_color = (1-(iter_num/total_iter_num), iter_num/total_iter_num, 0)
            
            plt.plot(x_axis, norm.pdf(x_axis, mu, sigma), color=curve_color)
            iter_num += 1

        plt.show()

if __name__ == "__main__":
    main()
