import matplotlib.pyplot as plt 
import gzip, os, sys
import numpy as np
from scipy.stats import multivariate_normal

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

# Function that downloads a specified MNIST data file from Yann Le Cun's website
def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

# Invokes download() if necessary, then reads in images
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1,784)
    return data

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Show a single MNIST digit. To do this, it first has to reshape the
# 784-dimensional vector into a 28x28 image.
def displaychar(image):
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

## Load the training set
train_data = load_mnist_images('train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')

## Load the testing set
test_data = load_mnist_images('t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
#train_data.shape, train_labels.shape


def fit_generative_model(x,y):
    """ This function takes a training set (data x and labels y) and fits a Gaussian generative model.
    It should return the parameters of this generative model; for each label j = 0,1,...,9, we have:
        - pi[j]: the frequency of that label
        - mu[j]: the 784-dimensional mean vector
        - sigma[j]: the 784x784 covariance matrix
    This means that pi is 10x1, mu is 10x784, and sigma is 10x784x784.
    """
    k = len(np.unique(y)) # labels 0,1,...,k-1
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k,d))
    sigma = np.zeros((k,d,d))
    pi = np.zeros(k)
    ###
    ### Your code goes here
    ###
    for label in range(0,k):
        indices = (y == label)
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1)
        pi[label] = float(sum(indices))/float(len(y))
    # Halt and return parameters
    return mu, sigma, pi

mu, sigma, pi = fit_generative_model(train_data, train_labels)
#displaychar(mu[0])
#displaychar(mu[1])
#displaychar(mu[2])

# Make predictions on test data
# Compute log Pr(label|image) for each [test image,label] pair.
k = len(np.unique(test_labels)) # labels 0,1,...,k-1
score = np.zeros((len(test_labels),k))
for label in range(0,k):
    rv = multivariate_normal(mean=mu[label], cov=sigma[label], allow_singular=True)
    for i in range(0,len(test_labels)):
       score[i,label] = np.log(pi[label]) + rv.logpdf(test_data[i,:])
predictions = np.argmax(score, axis=1)
# Finally, tally up score
errors = np.sum(predictions != test_labels)
print("Your model makes " + str(errors) + " errors out of 10000")
