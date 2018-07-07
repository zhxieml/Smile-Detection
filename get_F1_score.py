import time
from sklearn import svm, preprocessing
import random
import numpy as np
import os
from sklearn.externals import joblib
import sys


def view_bar(num, mes):
    rate_num = num
    number = int(rate_num / 4)
    hashes = '=' * number
    spaces = ' ' * (25 - number)
    r = "\r\033[31;0m%s\033[0m：[%s%s]\033[32;0m%d%%\033[0m" % (mes, hashes, spaces, rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


lbp_file = open('result_output.txt', 'r')
lbp_features = lbp_file.readlines()
lbp_file.close()
lbp_list = []
a = 0
for line in lbp_features:
    line = line.strip()
    line = eval(line)
    lbp_list.append(line)
    a += 1
    view_bar(a / 3884 * 100, 'LBP Processing')
print("\nLbp has been loaded!")

label_file = open('real_labels.txt', 'r')
labels = label_file.readlines()
label_file.close()
label_list = []
for label in labels:
    label = label.strip()
    label_list.append(int(label))
print("Label has been loaded!")

xs = list(range(3880))
rng = random.Random()
rng.shuffle(xs)
fold = [xs[388 * i:388 * (i + 1)] for i in range(10)]
scores = []
for i in range(10):
    start = time.clock()
    print('Group' + str(i))
    path = fold[i]
    opath = [k for k in xs if k not in path]

    training = [lbp_list[i] for i in opath]
    training = np.array(training)
    label = [label_list[i] for i in opath]
    label = np.array(label)
    #  scaler = preprocessing.StandardScaler()
    #  training = scaler.fit_transform(training)

    #  parameters = {"kernel": ("linear", "rbf"), "C": list(range(1, 100))}
    #  svr = svm.SVC()
    #  clf = grid_search.GridSearchCV(svr, parameters)
    #  clf.fit(training, label)
    #  print(clf.best_params_)
    clf = svm.SVC(kernel='poly', C=4000)
    clf.fit(training, label)
    print("Fit successfully!")

    result = []
    for t in path:
        test = [lbp_list[t]]
        result.append(clf.predict(test))

    tp = 0
    fp = 0
    fn = 0
    for k in range(len(result)):
        if result[k] == [1] and path[k] <= 2132:
            tp += 1
        if result[k] == [1] and path[k] > 2132:
            fp += 1
        if result[k] == [0] and path[k] <= 2132:
            fn += 1
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f1)
    scores.append(f1)
    print(time.clock() - start)
    name = "train_model" + str(i) + ".m"
    joblib.dump(clf, name)

best_score = max(scores)
total = 0
for score in scores:
    total += score
ave_score = total / 10
print("The average F1 score is " + str(ave_score))
print("The highest F1 score is " + str(best_score))
i = scores.index(best_score)
print("The best model is model" + str(i))
isexist = os.path.exists('train_model.m')
if isexist:
    os.remove('train_model.m')
for k in range(10):
    if k != i:
        os.remove("train_model" + str(k) + ".m")
    else:
        os.rename("train_model" + str(k) + ".m", 'train_model.m')

