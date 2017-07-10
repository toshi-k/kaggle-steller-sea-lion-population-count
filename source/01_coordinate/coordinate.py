
import os
import time

import numpy as np
import pandas as pd

from collections import OrderedDict

import cv2

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

# ------------------------------
# main
# ------------------------------

tic = time.time()

if not os.path.exists('_img'):
    os.makedirs('_img')

dir_input = '../../input'
if not os.path.exists(dir_input):
    os.makedirs(dir_input)

mismatched = [3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234, 242, 268, 290, 311, 331, 344, 380,
              384, 406, 421, 469, 475, 490, 499, 507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712,
              721, 767, 779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905, 909, 913, 927, 946]

train_csv = pd.read_csv('../../dataset/Train/train.csv')

# load label colors ------------

palette = pd.read_csv('../../dataset/palette.csv')

colors_and_labels = OrderedDict((('red', 'adult_males'),
                                 ('magenta', 'subadult_males'),
                                 ('brown', 'adult_females'),
                                 ('blue', 'juveniles'),
                                 ('green', 'pups')))

color_means = palette.groupby('color').mean()
print(color_means)

# set parameters for detector --

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0.5

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.9

detector = cv2.SimpleBlobDetector(params)

mismatch_under = 0
mismatch_over = 0

coordinates_all = dict()
train_csv_str = list()

for target_num in train_csv.train_id.tolist():

    print('\ntarget image: ' + str(target_num) + ' --------------------')

    if target_num in mismatched:
        continue

    # make mask image -----------

    traindotted = cv2.imread('../../dataset/TrainDotted/{0:d}.jpg'.format(target_num))
    tarindotted = traindotted.astype('float64')

    train = cv2.imread('../../dataset/Train/{0:d}.jpg'.format(target_num))
    train = train.astype('float64')

    assert traindotted.shape[0] == train.shape[0], 'image height is different'
    assert traindotted.shape[1] == train.shape[1], 'image width is different'

    black = (traindotted.sum(2) < 10.0).astype('float64')
    black = cv2.blur(black, (10, 10))
    black = black > 0.1

    black = np.expand_dims(black, axis=2)
    black = np.tile(black, (1, 1, 3))

    traindotted[black] = train[black]
    # cv2.imwrite('_img/filled.png', traindotted)

    diff = traindotted - train
    diff = (diff * diff).sum(2)

    diff = cv2.blur(diff, (5, 5))

    threshold = 1000.0
    mask = (diff > threshold).astype('float64')
    # cv2.imwrite('_img/mask.png', mask * 255.0)

    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))

    img = cv2.imread('../../dataset/TrainDotted/{0:d}.jpg'.format(target_num))
    img = img[:, :, ::-1]

    # apply mask ----------

    img = img * mask
    # cv2.imwrite('_img/masked.png', img * 255.0)

    traindotted = traindotted[:, :, ::-1]

    result = np.copy(traindotted)

    num_key_points_estimated = list()
    num_key_points_expected = train_csv.loc[target_num, :].drop('train_id')

    # detect key points for each labels ----------

    coordinates = dict()

    for color_target, label_target in colors_and_labels.iteritems():

        # print('target: ' + label_target + ' (' + color_target + ')')

        means = color_means.loc[color_target, :]
        means_tiled = np.tile(means, (img.shape[0], img.shape[1], 1))

        diff = img - means_tiled
        diff = diff * diff

        diff = np.sum(diff, axis=2)

        diff -= np.min(diff)
        diff /= np.max(diff)

        diff *= 255.0
        diff = diff.astype('uint8')

        # binarize by threshold ----------

        threshold = 13.0
        cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY, diff)

        # detect key points ----------

        key_points = detector.detect(diff)

        if num_key_points_expected[label_target] == 0 and len(key_points) != 0:
            print('\tnumber of points should be zero !')
            print('\t' + str(len(key_points)) + ' -> 0')
            key_points = list()

        num_key_points = len(key_points)

        if num_key_points_expected[label_target] > num_key_points:
            mismatch_under += num_key_points_expected[label_target] - num_key_points
        elif num_key_points_expected[label_target] < num_key_points:
            mismatch_over += num_key_points - num_key_points_expected[label_target]

        num_key_points_estimated.append(num_key_points)

        coordinate = list()
        for key_point in key_points:
            x, y = key_point.pt
            coordinate.append([x, y])

        coordinates.update({label_target: coordinate})

        for i in range(len(key_points)):
            key_points[i].size *= 10

        color_tuple = tuple(color_means.loc[color_target])
        result = cv2.drawKeypoints(result, key_points, np.array([]), color_tuple,
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('_img/result_{0:d}.png'.format(target_num), result[:, :, ::-1])

    diff = int(np.sum(np.abs(num_key_points_expected - np.array(num_key_points_estimated))))

    print('expected: [' + ', '.join(['{0:3d}'.format(s) for s in num_key_points_expected.tolist()]) + '] ' +
          'estimated: [' + ', '.join(['{0:3d}'.format(s) for s in num_key_points_estimated]) + '] ' +
          'diff: {0:d}'.format(diff))

    train_csv_line = str(target_num) + ',' + ','.join(str(s) for s in num_key_points_estimated) + '\n'
    train_csv_str.append(train_csv_line)
    coordinates_all.update({str(target_num): coordinates})

print('\n complete !')

# write result to files --------

f = open(os.path.join(dir_input, 'train_csv_v1.csv'), 'w')
f.write('train_id,adult_males,subadult_males,adult_females,juveniles,pups\n')
f.writelines(train_csv_str)
f.close()

f = open(os.path.join(dir_input, 'coordinates_v1.json'), "w")
json.dump(coordinates_all, f)
f.close()

# display summary of result ----

computational_time = (time.time() - tic) / 60.0
print('computational time: {0:.0f} [min]'.format(computational_time))

print('mismatch (number of under count) : {0:d}'.format(mismatch_under))
print('mismatch (number of over count)  : {0:d}'.format(mismatch_over))
print('mismatch (all)                   : {0:d}'.format(mismatch_under + mismatch_over))

# 857 have some trouble
# computational time: 84 [min]
# mismatch (number of under count) : 1046
# mismatch (number of over count)  : 266
# mismatch (all)                   : 1312
