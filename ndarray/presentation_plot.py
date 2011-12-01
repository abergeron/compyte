import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


colors = ('#e0ffff', '#ffc0cb')
sizes = (5, 5, 5)
csize = sizes[1] * sizes[2]

def print_save(filename):
    if True:
        plt.savefig(filename)
        print "saved in file", filename
        plt.cla()
        plt.clf()
    else:
        plt.show()

if False:
    # 3d dot 1 color
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #x = np.linspace(0, 1, 100)
    #y = np.sin(x * 2 * np.pi) / 2 + 0.5
    #ax.plot(x, y, zs=0, zdir='z', label='3d cube')

    for i in range(sizes[0]):
        for j in range(sizes[1]):
            for k in range(sizes[2]):
                #ax.scatter([i]*10, [j]*10, range(10),# marker='o',
                 #      linewidths=None, c=colors[1], alpha=1)
                ax.scatter(i, j, k)

    ax.set_title("3d dot")
    ax.legend()
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, sizes[1])
    ax.set_zlim3d(0, sizes[2])

    plt.show()

if False:
    # 3d dot 2 color
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #x = np.linspace(0, 1, 100)
    #y = np.sin(x * 2 * np.pi) / 2 + 0.5
    #ax.plot(x, y, zs=0, zdir='z', label='3d cube 2 colors')

    for i in range(10):
        c = colors[i % 2]
        for j in range(10):
            for k in range(10):
                ax.scatter(i, j, k, c=c)

    ax.legend()
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, sizes[1])
    ax.set_zlim3d(0, sizes[2])

    plt.show()


if False:
    # 2d dot 2 color
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(10):
        c = colors[i % 2]
        ax.plot([i] * csize, range(csize), c=c)
#    ax.plot(x, y)#, zs=0, zdir='z', label='3d cube 2 colors')
    ax.set_title("Reshaped as 2d")
#    matplotlib.rcParams['axes.unicode_minus'] = False

    ax.legend()
    ax.set_xlim(0, sizes[0])
    ax.set_ylim(0, csize)

    plt.show()


if False:
    # 3d dot reshaped to 2d 2 colors
    fig = plt.figure()
    ax = fig.gca(projection='3d')

#    x = np.linspace(0, 1, 100)
#    y = np.sin(x * 2 * np.pi) / 2 + 0.5
#    ax.plot(x, y, zs=0, zdir='z', label='3d cube')

    for i in range(10):
        c = colors[i % 2]
        ax.plot([i] * csize, range(csize), 'o', c=c)

    ax.legend()
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, csize)

    plt.show()

if False:
    # 3d dot sliced reshaped
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #x = np.linspace(0, 1, 100)
    #y = np.sin(x * 2 * np.pi) / 2 + 0.5
    #ax.plot(x, y, zs=0, zdir='z', label='3d cube')

    for i in range(sizes[0]):
        c = colors[i % 2]
        for j in range(csize):
            ax.scatter(i, j, c=c)

    ax.legend()
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, csize)

    ax.set_title("3d hypercube sliced reshaped")
    plt.show()


if False:
    # 3d plane 1 color
    fig = plt.figure()
    ax = Axes3D(fig)

    X, Y, Z = np.mgrid[0:sizes[0]:sizes[0] * 1j,
                       0:sizes[1]:10j,
                       0:sizes[2]:10j]
    ax = fig.gca(projection='3d')
    for i in range(X.shape[0]):
        cset = ax.contour(X[i], Y[0], Z[0], extend3d=True, colors="r")
    ax.set_title("3d hypercube")
    print_save("3d_hypercude.pdf")


if True:
    # 3d plane 2 color
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.shade = False
    X, Y, Z = np.mgrid[0:sizes[0]:sizes[0] * 1j,
                       0:sizes[1]:10j,
                       0:sizes[2]:10j]
    ax = fig.gca(projection='3d')
    for i in range(X.shape[0]):
        cset = ax.contour(X[i], Y[0], Z[0], extend3d=True,
                          colors=colors[i % 2])
    ax.set_title("3d hypercube sliced")
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, csize)
    ax.set_zlim3d(0, csize)
    print_save("3d_hypercude_sliced.pdf")


if True:
    # 3d plane 2 color reshaped
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.shade = False

    X, Y, Z = np.mgrid[0:sizes[0]:sizes[0] * 1j,
                       0:csize:10j,
                       0:1:10j]
    ax = fig.gca(projection='3d')
    ax.shade=False
    for i in range(X.shape[0]):
        cset = ax.contour(X[i], Y[0], Z[0], extend3d=True,
                          colors=colors[i % 2])
    ax.set_title("3d hypercube sliced reshaped")
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, csize)
    ax.set_zlim3d(0, csize)
    print_save("3d_hypercude_sliced_reshaped.pdf")


if  False:  # True:
    # some weird test
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    #x = np.linspace(0, 1, csize)
    #y = np.sin(x * 2 * np.pi) / 2 + 0.5
    #ax.plot(x, y, zs=0, zdir='z', label='3d cube')

    for i in range(sizes[0]):
        c = colors[i % 2]
        for j in range(csize):
            ax.scatter(i, j, c=c)

    ax.legend()
    ax.set_xlim3d(0, sizes[0])
    ax.set_ylim3d(0, csize)

    ax.set_title("3d hypercube sliced reshaped")
    plt.show()
