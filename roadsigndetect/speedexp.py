import argparse
from matplotlib import pyplot as plt
import matplotlib
import importlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from play import *

def foo(opts):
    i = opts['i']
    j = opts['j']
    return (i*j, i+j)

def exp(opts):
    opts['mode'] = 'trainspeed'
    rsegs = range(1,6,1) 
    csegs = range(1,10,1) 
    nr = len(rsegs)
    nc = len(csegs)
    rsegs, csegs = np.meshgrid(rsegs, csegs, sparse=False, indexing='ij')
    rsegs = rsegs.astype(np.int32)
    csegs = csegs.astype(np.int32)
    mses = np.empty_like(rsegs, dtype=np.float32)
    vars = np.empty_like(rsegs, dtype=np.float32)

    inputs = []
    for i in range(nr):
        for j in range(nc):
            cp = opts.copy()
            cp['rseg'] = rsegs[i,j]
            cp['cseg'] = csegs[i,j]
            cp['i'] = i 
            cp['j'] = j 
            inputs.append(cp)

    pool = ThreadPool(opts['numthread'])
    tic()
    # results[0] = trainModel(inputs[0])
    results = pool.map(trainModel, inputs, 1)
    # results = pool.map(foo, inputs, 1)
    pool.close()
    pool.join()
    toc()
    for i, res in enumerate(results):
        inp = inputs[i]
        i = inp['i']
        j = inp['j']
        mse, var = res
        mses[i,j] = mse
        vars[i,j] = var
    pickle.dump(rsegs , open('{0}/{1}.p'.format(SCRATCH_PATH, "rsegs"), "wb" ))
    pickle.dump(csegs , open('{0}/{1}.p'.format(SCRATCH_PATH, "csegs"), "wb" ))
    pickle.dump(mses  , open('{0}/{1}.p'.format(SCRATCH_PATH, "mses" ), "wb" ))
    pickle.dump(vars  , open('{0}/{1}.p'.format(SCRATCH_PATH, "vars" ), "wb" ))

def plot():
    rsegs = pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "rsegs.p"), "rb" ))
    csegs = pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "csegs.p"), "rb" ))
    mses  = pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "mses.p") , "rb" ))
    vars  = pickle.load(open('{0}/{1}'.format(SCRATCH_PATH, "vars.p") , "rb" ))

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    #ax.plot_surface(rsegs, csegs, mses, color='b')
    ax.plot_wireframe(rsegs, csegs, mses)

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(rsegs, csegs, vars, rstride=4, cstride=4, color='b')
    fig.set_size_inches(14, 6)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', 
            default='{0}2011_09_26-2/data'.format(KITTI_PATH),
            help='Specify path for the image files')
    parser.add_argument('--delay', dest='delay', nargs='?', default=0.05, type=float,
            help='Amount of delay between images')
    parser.add_argument('--start-frame', dest='startframe', nargs='?', default=0, type=int,
            help='Starting frame to play')
    parser.add_argument('--end-frame', dest='endframe', nargs='?', default=-1, type=int,
            help='Ending frame to play, -1 for last frame')
    parser.add_argument('--num-frame', dest='numframe', nargs='?', default=-1, type=int,
            help='Number of frame to play, -1 for all frames')
    parser.add_argument('--mode', dest='mode', action='store', default='plot')
    parser.add_argument('--rseg', dest='rseg', nargs='?', default=3, type=int,
            help='Number of vertical segmentation in computing averaged flow')
    parser.add_argument('--cseg', dest='cseg', nargs='?', default=4, type=int,
            help='Number of horizontal segmentation in computing averaged flow')
    parser.add_argument('--num-thread', dest='numthread', nargs='?', default=4, type=int,
            help='number of thread to run training')

    (opts, args) = parser.parse_known_args()

    if (opts.mode=='train'):
        exp(vars(opts))
    elif (opts.mode=='plot'):
        plot()

if __name__ == "__main__":
    main()
