# Example
# Same example as above, but return a 2-D array with 3 rows, each containing 5 column values.
import numpy as np
import matplotlib.pyplot as plt
x = np.random.choice([1,3,5,7], p=[0.1,0.3,0.5,0.1], size=(3,5))
y = np.random.choice([5,6,7,8], p=[0.2,0.5,0.1,0.2], size=(3,5))
z = np.array([1,2,3,4])
k = np.array([6,7,8,9])
print(x)
plt.boxplot(x)
#plt.hist(x)
#plt.plot(x)
#plt.contour(x)
#plt.scatter(x,y)
#plt.eventplot(x)
#plt.prism()
#plt.stackplot(z,k)
plt.show()


# Random Permutations of Elements
# A permutation refers to an arrangement of elements. e.g. [3, 2, 1] is a permutation
# of [1, 2, 3] and vice-versa.
# The NumPy Random module provides two methods for this: shuffle() and permutation()
# Shuffling Arrays
# Shuffle means changing arrangement of elements in-place. i.e. in the array itself.
from numpy import random
arr = np.array([1,2,3,4,5,6])
random.shuffle(arr)
print(arr) # The shuffle() method makes changes to the original array.


# Permutations
# Generate a random permutation of elements of following array:
from numpy import random
arr = np.array([1,2,3,4,5])
newarr = random.permutation(arr)
print(arr)
print()
print(newarr)   # The permutation() method returns a re-arranged array
                # (and leaves the original array un-changed).

# Seaborn
# Seaborn is a library that uses Matplotlib underneath to plot graphs.
# It will be used to visualize random distributions.

#Distplots
# Distplot stands for distribution plot, it takes as input an array and plots
# a curve corresponding to the distribution of points in the array.
from matplotlib import pyplot as plt
import seaborn as sns

sns.distplot([1,2,3,4,5])
plt.show() # This displays histogram of the array


# Plotting a Distplot Without the Histogram
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([0,1,2,3,4,5,], hist=False)
plt.show()


# Normal Distribution
# The Normal Distribution is one of the most important distributions.
#
# It is also called the Gaussian Distribution after the German mathematician Carl Friedrich Gauss.
#
# It fits the probability distribution of many events, eg. IQ Scores, Heartbeat etc.
#
# Use the random.normal() method to get a Normal Data Distribution.
#
# It has three parameters:
#
# loc - (Mean) where the peak of the bell exists.
#
# scale - (Standard Deviation) how flat the graph distribution should be.
#
# size - The shape of the returned array.
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
x = random.normal(size=(2,3))
print(x)
sns.distplot(x)
plt.show()


# Generate a random normal distribution of size 2x3 with mean at 1 and standard deviation of 2:
from numpy import random
y = random.normal(loc=1, scale=2, size=(2,3))
print(y)
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(y)
plt.show()

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=1, scale=2, size=1000),hist=False)
plt.xlabel('Data Size')
plt.ylabel('Accuracy')
plt.show()

#Visualization of Normal Distribution
import matplotlib.pyplot as plt
from numpy import random
import seaborn as sns

sns.distplot(random.normal(size=(10,5)),hist=False,label="Generalization Error")
plt.show()


# Binomial Distribution
# Binomial Distribution is a Discrete Distribution.
#
# It describes the outcome of binary scenarios, e.g. toss of a coin, it will either be
# head or tails.
#
# It has three parameters:
#
# n - number of trials.
#
# p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).
#
# size - The shape of the returned array.

# NOTE: Discrete Distribution:The distribution is defined at separate set of events,
# e.g. a coin toss's result is discrete as it can be only head or tails whereas height
# of people is continuous as it can be 170, 170.1, 170.11 and so on.

# Given 10 trials for coin toss generate 10 data points:
from numpy import random
x = random.binomial(n=10, p=0.5, size=10)
print(x)


# Visualization of Binomial Distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.binomial(n=10, p=0.5, size=1000),hist=True, kde=False)
plt.show()


# Difference Between Normal and Binomial Distribution
# The main difference is that normal distribution is continous whereas binomial is discrete,
# but if there are enough data points it will be quite similar to normal distribution
# with certain loc and scale.

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='Normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='Binomial')
plt.show()


#Poisson Distribution is a Discrete Distribution.

#It estimates how many times an event can happen in a specified time. e.g. If someone eats twice a day
# what is probability he will eat thrice?

#It has two parameters:

#lam - rate or known number of occurences e.g. 2 for above problem.

#size - The shape of the returned array.

from numpy import random
x = random.poisson(lam=2,size=10)
print(x)


#Visualization of Poisson Distribution
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.poisson(lam=2, size=1000),kde=False)
plt.show()


# Difference Between Normal and Poisson Distribution
# Normal distribution is continous whereas poisson is discrete.

# But we can see that similar to binomial for a large enough poisson distribution
# it will become similar to normal distribution with certain std dev and mean.
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.normal(loc=50, scale=7, size=1000),hist=False, label='Normal')
sns.distplot(random.poisson(lam=50, size=1000),hist=False,label='Poisson')
plt.show()


# Difference Between Poisson and Binomial Distribution
# The difference is very subtle it is that, binomial distribution is for discrete trials,
# whereas poisson distribution is for continuous trials.

# But for very large n and near-zero p binomial distribution is near identical
# to poisson distribution such that n * p is nearly equal to lam.

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(random.binomial(n=1000, p=0.01,size=1000), hist=False, label='Binomial')
sns.distplot(random.poisson(lam=10,size=1000),hist=False, label='Poisson')
plt.show()
