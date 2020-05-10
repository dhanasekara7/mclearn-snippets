```python
# jupyter inline 
# no zoom
%matplotlib inline

# interactivity
# not work in jupyter lab
%matploylib notebook

# works in jupyter lab
# need to install plugin "jupyter-matplotlib"
%matplotlib widget

# figure and axes
# figure = one window/image
# axe = area with in figure

# create single axes
fig = plt.figure()

fig, ax = plt.subplots(2, 2)


# ax = plt.subplot(n, m, i)
ax11 = plt.subplot(2, 2, 1)
ax21 = plt.subplot(2, 2, 2)
ax12 = plt.subplot(2, 2, 3)
ax22 = plt.subplot(2, 2, 4)

# equivalent
fig, axes = plt.subplots(2, 2)
ax11, ax21, ax12, ax22 = axes.ravel())

#sample ( stateful interface )
sin = np.sin(np.linspace(-4, 4, 100))
plt.subplot(2, 2, 1)
plt.plot(sin)
plt.subplot(2, 2, 2)
plt.plot(sin, c='r')

# equivalent ( object oriented interface )
fig, axes = plt.subplots(2)
axes[0, 0].plot(sin)
axes[0, 1].plot(sin, c='r')


# some more comparisions
plt.title => ax.set_title
plt.xlim, plt.ylim => ax.set_xlim, ax.set_ylim
plt.xlabel, plt.ylabel => ax.set_xlabel, ax.set_ylabel 
plt.xticks, plt.yticks => ax.set_xticks, ax.set_yticks
                          (& ax.set_xtick_labels )

ax = plt.gca() # get current axes
fig = plt.gcf() # get current figure


```