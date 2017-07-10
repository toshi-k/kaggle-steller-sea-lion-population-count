
------------------------------
-- library
------------------------------

local json = require('json')
require 'lib/load_train_list'
require 'lib/load_mismatch_list'
require 'lib/setdiff'
require 'lib/split_valid'
require 'lib/load_train_csv'
require 'lib/load_test_list'
require 'lib/gen_coordinate_positive'

------------------------------
-- main
------------------------------

print('==> load data')

labels = {'adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups'}
patch_size = 224

-- load coordinate -----------

coordinate = json.load('../../input/coordinates_v1.json')

-- load train list -----------

local all_list_num = load_train_list('../../dataset/TrainDotted')

-- load mismatch list --------

local mismatch_list = load_mismatch_list('../../dataset/MismatchedTrainImages.txt')

-- generate match list -------

local match_list = setdiff(all_list_num, mismatch_list)

-- split train and valid -----

train_list, valid_list = split_valid(match_list)

-- load train.csv ------------

train_csv = load_train_csv('../../dataset/Train/train.csv')

-- load test list ------------

test_list = load_test_list('../../dataset/sample_submission.csv')

-- calc where sea lions exist-

coordinate_positive = gen_coordinate_positive(coordinate, train_list)
