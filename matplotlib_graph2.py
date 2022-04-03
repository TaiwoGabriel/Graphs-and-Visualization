
# Naturally, data scientists want a
# way to visualize their data. Either they are wanting to see it for themselves to get a better
# grasp of the data, or they want to display the data to convey their results to someone. With
# Matplotlib, arguably the most popular graphing and data visualization module for Python,
# this is very simplistic to do
# Import matplotlib library
import matplotlib.pyplot as plt

x = [1,2,3]
y = [4,5,1]
# Ploting on the canvas
plt.plot(x,y) # the plot() will give a line graph
#plt.scatter(x,y)
plt.title('Epic Movie')
plt.xlabel('Number of movies')
plt.ylabel('Users')
# Showing what we have plotted
plt.show()


# Using Styles and Linewidth
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

# Can plot specifically, after just showing the defaults
plt.plot(x,y,linewidth=5)
plt.plot(x2,y2,linewidth=5)
plt.title('Epic Info')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
plt.show()


# Using legend() grid() and color
# Using Styles and Linewidth
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

# Can plot specifically, after just showing the defaults
plt.plot(x,y,'g',linewidth=5,label='Line One')
plt.plot(x2,y2,'c',linewidth=5, label='Line Two')
plt.title('Epic Info')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')

plt.legend() # legend() displays the values assigned to the label()
plt.grid(True,color='k')

plt.show()


# Bar chats
# Bar charts with matplotlib are basically 1 slight change, same with scatter plots.
# The only major change I like to make to bar charts is to center them, and that's about it:
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

plt.bar(x,y,align='center')
plt.bar(x2,y2,align='center', color='g')
plt.title('Epic Info')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
plt.show()


# Scatter Plot
# How about scatter plots? Super easy, we'll just change .bar() to .scatter(), and remove our align parameter:
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

x = [5,8,10]
y = [12,16,6]

x2 = [6,9,11]
y2 = [6,15,7]

plt.scatter(x,y)
plt.scatter(x2,y2,color='g')
plt.title('Epic Info')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
plt.show()


# Plotting Dataset
# Eventually, you'll probably find that people stop using CSV files and use either databases
# or they are using something like HDF5 formatting. For now, let's just cover CSV.
# There are obviously many ways to read files in Python. You can use Python's CSV module
# that is a part of the standard library. You can make use of Numpy's loadtxt as well,
# which we'll be using. Another fantastic choice is using Pandas! So there are many choices to consider,
# but, for now, we're going to use Numpy. Depending on your goals and requirements, you might eventually
# wind up choosing something else. I like NumPy because it's very open-ended for data analysis,
# yet still very powerful. I also think Pandas is going to be a great choice for most people,
# but it is less open-ended.


#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#df = pd.read_csv('exampleFile.csv')
#print(df)
x,y = np.loadtxt('exampleFile.csv',
                 unpack=True,
                 delimiter=',')
plt.plot(x,y)
plt.title('Epic Info')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
plt.show()

# Here, our major new things are importing numpy, and then using numpy's loadtxt function.
# Loadtxt can be used to load more than just .txt files. It's just load things with text, that's all.
# Here, we are unpacking the contents of exampleFile.csv, using the delimiter of a comma.
# It's important to note here that you MUST unpack the exact same number of columns that will come
# from the delimiter that you state. If not, you'll get an error.


# import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np

# Prepare the data
x = np.linspace(0,10,100)

# Plot the data
plt.plot(x,x, label='Linear')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# Figure
# To the figure you add Axes. The Axes is the area on which the data is plotted with
# functions such as plot() and scatter() and that can have ticks, labels, etc. associated with it.
# This explains why Figures can contain multiple Axes.
#
# Tip: when you see, for example, plt.xlim, you’ll call ax.set_xlim() behind the covers.
# All methods of an Axes object exist as a function in the pyplot module and vice versa.
# Note that mostly, you’ll use the functions of the pyplot module
# because they’re much cleaner, at least for simple plots!
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3,4],[10,20,25,30], color='lightblue',linewidth=3)
ax.scatter([0.3, 3.8, 1.2, 2.5],[11, 25, 9, 26],color='darkgreen',marker='^')
ax.set_xlim(0.5, 4.5)
plt.show()

# The code above can be written in another simple way as given below


import matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4],[10,20,25,30],color='lightblue',linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5],[11, 25, 9, 26], color='darkgreen',marker='^')
plt.xlim(0.5, 4.5)
plt.show()

# The second code chunk is definitely cleaner, isn’it it?
# However, if you have multiple axes, it’s still better to make use of the first code chunk because
# it’s always better to prefer explicit above implicit code! In such cases, you want to make use of the Axes object ax.


# Creating your Plots
# Alright, you’re off to create your first plot yourself with Python!
# As you have read in one of the previous sections, the Figure is the first step
# and the key to unlocking the power of this package. Next, you see that you initialize the axes of
# the Figure in the code chunk above with fig.add_axes():

#importing pyplot
import matplotlib.pyplot as plt

# initialize a figure
fig = plt.figure()

# Add axes to the figure
fig.add_axes([0,0,1,1])


# Creating subplots
# ou use subplots to set up and place your Axes on a regular grid.
# So that means that in most cases, Axes and subplot are synonymous,
# they will designate the same thing. When you do call subplot to add Axes to your figure,
# do so with the add_subplots() function. There is, however, a difference between the add_axes()
# and the add_subplots() function

import matplotlib.pyplot as plt
import numpy as np

#create a figure
fig = plt.figure()

# set up axes
ax = fig.add_subplot(2,2,1) # 111 is equal to 1,1,1, which means that you actually give three arguments
# to add_subplot(). The three arguments designate the number of rows (1), the number of columns (1)
# and the plot number (1). So you actually make one subplot
# 2,2,1 means you split the canvas into two rows, two columns and the plot number is 1. Your Figure will have
# four axes in total, arranged in a structure that has two rows and two columns.
# With the line of code that you have considered, you say that the variable ax is the first of the four axes
# to which you want to start plotting. The “first” in this case means that it will be the first axes on the
# left of the 2x2 structure that you have initialized.

# Scatter the data
ax.scatter(np.linspace(0,1,5), np.linspace(0,5,5))

# show the plot
plt.show()

# you can also use subplots() if you want to get one or more subplots at the same time.


# Creating two subplots and changing the size of Figures
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5)) # 20 means width and 10 represents height of the figure
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# # or replace the three lines of code above by the following line:
# #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10)

#plot the data
plt.bar([1,2,3],[3,4,5])
plt.barh([0.5,1,2.5],[0,1,2])
# show the plot
plt.show()


# Creating three subplot
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
x = [1,2,3,4]
y =[0.5,1,2,2.5]

# Plot the data
ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2])
ax2.axhline(0.45)
ax1.axvline(0.65)
ax3.scatter(x,y)

# Show the plot
plt.show()










# we cover how to plot multiple subplots on the same figure in Python's
# Matplotlib. The way the subplot numbers work can be somewhat confusing at first, but should
# be fairly easy to get the hang of. Later on, I will also show another way to modify the showing
# of multiple subplots, but this is the easiest way.
# Note: Things like legends are drawn when you call them, so, if you are using,
# say, subplots, and call legends at the very end, only the 2nd subplot would have a legend.
# If you wanted a legend on each subplot, then you would need to call it per subplot.
# This is the same with titles! But hey, I didn't even cover subplots
# (multiple graphs on the same "figure," which just means the same window)

import matplotlib.pyplot as plt
x = []
y = []

fig = plt.figure()
rect = fig.patch
rect.set_facecolor('#31312e')

readFile = open('exampleFile.txt', 'r')
sepFile = readFile.read().split('\n')
readFile.close()
for plotPair in sepFile:
    xAndY = plotPair.split(',')
    x.append(int(xAndY[0]))
    y.append(int(xAndY[1]))

ax1 = fig.add_subplot(2,2,1, axisbg='grey')
ax1.plot(x, y, 'c', linewidth=3.3)
ax1.tick_params(axis='x', colors='c')
ax1.tick_params(axis='y', colors='c')
ax1.spines['bottom'].set_color('w')
ax1.spines['top'].set_color('w')
ax1.spines['left'].set_color('w')
ax1.spines['right'].set_color('w')
ax1.yaxis.label.set_color('c')
ax1.xaxis.label.set_color('c')
ax1.set_title('Matplotlib graph', color = 'c')
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax2 = fig.add_subplot(2,2,2, axisbg='grey')
ax2.plot(x, y, 'c', linewidth=3.3)
ax2.tick_params(axis='x', colors='c')
ax2.tick_params(axis='y', colors='c')
ax2.spines['bottom'].set_color('w')
ax2.spines['top'].set_color('w')
ax3 = fig.add_subplot(2,1,2, axisbg='grey')
ax3.plot(x, y, 'c', linewidth=3.3)
ax3.tick_params(axis='x', colors='c')
ax3.tick_params(axis='y', colors='c')
ax3.spines['bottom'].set_color('w')
ax3.spines['top'].set_color('w')
ax3.spines['left'].set_color('w')
ax3.spines['right'].set_color('w')
ax3.yaxis.label.set_color('c')
ax3.xaxis.label.set_color('c')
ax3.set_title('Matplotlib graph', color = 'c')
ax3.set_xlabel('x axis')
ax3.set_ylabel('y axis')

plt.show()
