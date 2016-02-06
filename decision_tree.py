from __future__ import division

import string
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target

print("importing data array")
# import data array
learning_in = np.genfromtxt('letter-recognition_data.csv', delimiter=',')
learning_in = learning_in[1:]

# importing class array           
f = open('letter-recognition_class.csv')
ltr_lines = f.read().splitlines()
nmr_lines = np.zeros((len(ltr_lines)-1,1))

# importing attributes
f = open('letter-recognition_data.csv')
attr_lines = f.read().splitlines()
attr_lines = attr_lines[0].split(',')

# encoding classifications
idx = 1
for i in range(1,len(ltr_lines)-1):
    nmr_lines[i-1,0] = list(string.ascii_uppercase).index(ltr_lines[idx])
    idx += 1
enc = OneHotEncoder(sparse=False)       
learning_out = enc.fit_transform(nmr_lines)

# cross validation
X_train, X_test_split, Y_train, Y_test_split = cross_validation.train_test_split(learning_in, learning_out,test_size=.4,
                                                                                 random_state=0)
X_CV, X_test, Y_CV, Y_test = cross_validation.train_test_split(X_test_split, Y_test_split, test_size=.5,
                                                               random_state=0)

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
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train,
                                                                                         test_size=ind/X_train.shape[0],
                                                                                         random_state=0)

    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(X_train_step, Y_train_step)
    train_score.append(1 - clf_tree.score(X_train, Y_train))
    cv_score.append(1 - clf_tree.score(X_CV, Y_CV))
    test_score.append(1 - clf_tree.score(X_test, Y_test))
    depth.append(ind)

plt.figure()
plt.title('decision tree learning curve ')
plt.xlabel("training examples")
plt.ylabel("error")
plt.plot(depth, train_score, 'o-', color="r", label="training error")
#plt.plot(depth, cv_score, 'o-', color="g", label="Cross Validation error")
plt.plot(depth, test_score, 'o-', color="b",label="test error")
plt.legend(loc="best")
#plt.ylim(0,max(train_score))
#plt.show()
plt.savefig("dt_performance.png",bbox_inches='tight',dpi=100)
'''
# compute confusion matrix
def plot_confusion_matrix(cm, title='confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(list(string.ascii_uppercase)))
    plt.xticks(tick_marks, list(string.ascii_uppercase), rotation=45)
    plt.yticks(tick_marks, list(string.ascii_uppercase))
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

def OneHotDecoder(x):
    y = np.zeros((x.shape[0],1),dtype=int)
    
    x_int = x.astype(int)
    for j in range(0,x_int.shape[0]):
        x_str = map(str,x_int[j])
        temp = x_str[0]
        for i in range(1,len(x_str)):
            temp = temp + x_str[i]
        y[j] = int(temp,2)
        y[j] = math.log(y[j],2)
    return y;
    
Y_pred = clf_tree.fit(X_train,Y_train).predict(X_test_split)
y_test_cnt = OneHotDecoder(Y_test_split)
y_pred_cnt = OneHotDecoder(Y_pred)
cm = confusion_matrix(y_test_cnt,y_pred_cnt)
np.set_printoptions(precision=2)
print('confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='normalized confusion matrix')

plt.show()
'''
best_data_size = (cv_score.index(min(cv_score)) + 1) * 100

print(best_data_size)

cv_score = []
train_score = []
test_score = []
depth = []

for ind in range(1, 40):
    # slice the data
    X_discard, X_train_step, Y_discard, Y_train_step = cross_validation.train_test_split(X_train, Y_train,
                                                                                         test_size=best_data_size/X_train.shape[0],
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
plt.title('decision tree error and depth ')
plt.xlabel("tree depth")
plt.ylabel("error")
plt.plot(depth, train_score, 'o-', color="r", label="training error")
#plt.plot(depth, cv_score, 'o-', color="g", label="Cross Validation error")
plt.plot(depth, test_score, 'o-', color="b",label="test error")
plt.legend(loc="best")
#plt.ylim(0,max(train_score))
#plt.show()
plt.savefig("dt_depth.png",bbox_inches='tight',dpi=100)
print("done!")

