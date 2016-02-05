from __future__ import division

import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
#from scipy import stats

learning_in = np.genfromtxt('letter-recognition_data.csv', delimiter=',')
learning_in = learning_in[1::, :]
           
f = open('letter-recognition_class.csv')
ltr_lines = f.read().splitlines()
nmr_lines = np.zeros([len(ltr_lines)-1,1])
#nmr_lines = []
idx = 1
for i in range(1,len(ltr_lines)-1):
#    nmr_lines.append(list(string.ascii_uppercase).index(ltr_lines[idx]))
    nmr_lines[i-1,0] = list(string.ascii_uppercase).index(ltr_lines[idx])
    idx += 1
#nmr_lines.reshape(nmr_lines.shape[0],1)
enc = OneHotEncoder()       
learning_out = enc.fit_transform(nmr_lines)
#learning_out = learning_out[:, 0:-1]
#learning_in_norm = stats.zscore(learning_in, axis=1, ddof=1)
