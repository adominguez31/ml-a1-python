from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
from numpy import str
#from scipy import stats

learning_in = np.genfromtxt('letter-recognition_data.csv', delimiter=',')
learning_in = learning_in[1:-1, :]

enc = OneHotEncoder()
learning_out = enc.fit(np.genfromtxt('letter-recognition_class.csv', delimiter=','),dtype=np.str)
learning_out = learning_out[1:-1, :]

#learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)

X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out[:, 0],
                                                                                 test_size=.4, random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5, random_state=0)

print(X_train.shape)
print(X_CV.shape)
print(X_test.shape)
num_attrs = learning_in.shape[1]
cv_score = []
train_score = []
test_score = []
depth = []

for ind in range(100, X_train.shape[0], 100):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train,test_size=ind/X_train.shape[0], random_state=0)

    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score.append(1 - clf_tree.score(X_train, Y_train))
    cv_score.append(1 - clf_tree.score(X_CV, Y_CV))
    test_score.append(1 - clf_tree.score(X_test, Y_test))
    depth.append(ind)

plt.figure()
plt.title('Decision Tree Learing Curve ')
plt.xlabel("Training Examples")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.plot(depth, train_score, 'o-', color="r", label="Training error")
plt.plot(depth, cv_score, 'o-', color="g", label="Cross Validation error")
# plt.plot(depth, test_score, 'o-', color="b",label="Test error")
plt.legend(loc="best")
plt.ylim(0, .6)

best_data_size = (cv_score.index(min(cv_score)) + 1) * 100

print(best_data_size)

cv_score = []
train_score = []
test_score = []
depth = []

for ind in range(1, num_attrs):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train,
                                                                                         test_size=best_data_size /
                                                                                                   X_train.shape[0],
                                                                                         random_state=0)

    clf_tree = tree.DecisionTreeClassifier(max_depth=ind)
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score.append(1 - clf_tree.score(X_train, Y_train))
    cv_score.append(1 - clf_tree.score(X_CV, Y_CV))
    test_score.append(1 - clf_tree.score(X_test, Y_test))
    depth.append(ind)

best_depth = cv_score.index(min(cv_score)) + 1
print(best_depth)

plt.figure()
plt.title('Decision Tree Error and Depth ')
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.ylim(0, .6)
plt.plot(depth, train_score, 'o-', color="r", label="Training error")
plt.plot(depth, cv_score, 'o-', color="g", label="Cross Validation error")
# plt.plot(depth, test_score, 'o-', color="b",label="Test error")
plt.legend(loc="best")
plt.ylim(0, .6)
plt.show()