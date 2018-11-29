#
# UCSanDiegoX: DSE220x winery dataset
#

import numpy as np
import matplotlib.pyplot as plt
# Useful module for dealing with the Gaussian density
from scipy.stats import norm, multivariate_normal

# Load data set.
data = np.loadtxt('wine.data.txt', delimiter=',')
# Names of features
featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                'OD280/OD315 of diluted wines', 'Proline']
# Split 178 instances into training set (trainx, trainy) of size 130 and test set (testx, testy) of size 48
np.random.seed(0)
perm = np.random.permutation(178)
trainx = data[perm[0:130],1:14]
trainy = data[perm[0:130],0]
testx = data[perm[130:178], 1:14]
testy = data[perm[130:178],0]

"""
We now define a function that fits a Gaussian generative model to the data. For each class (j=1,2,3), we have:

pi[j]: the class weight
mu[j,:]: the mean, a 13-dimensional vector
sigma[j,:,:]: the 13x13 covariance matrix
This means that pi is a 4x1 array (Python arrays are indexed starting at zero, and we aren't using j=0), mu is a 4x13 array and sigma is a 4x13x13 array.
"""
def fit_generative_model(x,y):
    k = 3  # labels 1,2,...,k
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k+1,d))
    sigma = np.zeros((k+1,d,d))
    pi = np.zeros(k+1)
    for label in range(1,k+1):
        indices = (y == label)
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1)
        pi[label] = float(sum(indices))/float(len(y))
    return mu, sigma, pi

# Fit a Gaussian generative model to the training data
mu, sigma, pi = fit_generative_model(trainx,trainy)

# Now test the performance of a predictor based on a subset of features
def test_model(mu, sigma, pi, features, tx, ty):
    ''' Define a general purpose testing routine that takes as input:
        - the arrays pi, mu, sigma defining the generative model, as above
        - the test set (points tx and labels ty)
        - a list of features features (chosen from 0-12)
    It should return the number of mistakes made by the generative model
    on the test data, when restricted to the specified features.
    '''
    ###
    ### Your code goes here
    ###
    k = len(np.unique(ty))
    p = np.zeros([len(testy), k+1])
    for lable in range(1,k+1):
        rv = multivariate_normal(mu[lable][features], sigma[lable][features,features])
        p[:,lable] = rv.pdf(tx[:,features])
    predicts = np.argmax(p, axis=1)
    errors=(ty.astype(int) != predicts)
    return sum(errors)

for feat in [[2], [0,2], [0,2,6]]:
    errors = test_model(mu, sigma, pi, feat, testx, testy)
    fstr = ','.join([str(elem) for elem in feat])
    print('Test errors for feature %s:\t%d' %(fstr, errors))
print('Test errors for feature range(0,13):\t%d' %test_model(mu, sigma, pi, range(0,13), testx, testy))
