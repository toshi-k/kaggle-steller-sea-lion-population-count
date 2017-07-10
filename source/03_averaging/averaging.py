
import os
import re

import pandas as pd

# ------------------------------
# main
# ------------------------------

target_dir = '../../submission/submission_pre'
target_files = os.listdir(target_dir)

submission = pd.read_csv('../../dataset/sample_submission.csv')

columns = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']
scores = []

# averaging --------------------

for target in target_files:

    score = float(re.sub('.+_valid([0-9\.]+)\.csv', '\\1', target))
    scores.append(score)

    df = pd.read_csv(os.path.join(target_dir, target))
    # print(df.head())

    for column in columns:
        submission[column] += df[column]

# format result to integer -----

for column in columns:
    submission[column] /= len(target_files)
    submission[column] = submission[column].astype(int)

print(submission.head())

# save result ------------------

ave_score = sum(scores) / len(scores)
output_file = target_dir[:-4] + '_num_model' + str(len(target_files)) + '_mvalid{0:.2f}'.format(ave_score) + '.csv'
print('output: ' + output_file)

submission.to_csv(output_file, index=False)
