import argparse
from os import listdir
from os.path import isfile, join, splitext
from ntpath import basename
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
from path import *
import csv
from multiprocessing.pool import ThreadPool
from functools import partial
from sklearn import datasets, linear_model
        
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, bgr('g'))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 2, bgr('g'), -1)
    return img

def draw_avgflow(img, avgflow):
    h, w = img.shape[:2]
    hs, ws = avgflow.shape[:2]
    hstep = h/hs
    wstep = w/ws
    # print(h,w, hstep, wstep, hs, ws, h/hstep, w/wstep)
    y, x = np.mgrid[hstep/2:hstep*hs:hstep, wstep/2:wstep*ws:wstep].reshape(2,-1).astype(int)
    ys, xs = np.mgrid[0:hs, 0:ws].reshape(2,-1).astype(int)
    fx, fy = avgflow[ys,xs].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, bgr('r'), thickness=2)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 4, bgr('r'), -1)
    return img

def detflow(frame, prev, cur, **options):
    flowmode = options['flowmode']
    gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow, avgflow = getflow(prevgray, gray, **options)
    if flowmode == 'allflow':
        frame = draw_flow(frame, flow)
    elif flowmode == 'avgflow':
        frame = draw_flow(frame, flow)
        frame = draw_avgflow(frame, avgflow)
    return frame

def getflow(prevgray, gray, **options):
    rseg = options['rseg']
    cseg = options['cseg']

    # gray = np.round(np.random.rand(4,4,2)*3)
    h, w = gray.shape[:2]
    rstride = h / rseg
    cstride = w / cseg
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # flow = gray 
    avgflow = np.ndarray((rseg, cseg, 2), dtype=flow.dtype)
    for ir in range(0, rseg):
        rstart = ir*rstride
        rend = min(rstart+rstride, h)
        for ic in range(0, cseg):
            cstart = ic*cstride
            cend = min(cstart+cstride, w)
            grid = flow[rstart:rend, cstart:cend]
            avgflow[ir, ic] = np.mean(grid, axis=(0,1))
    return flow, avgflow

def loadHeader(path):
    headers = {}
    with open('{0}/dataformat.txt'.format(path), 'r') as dataformat:
        for i, line in enumerate(dataformat):
            headers[line.split(':')[0]] = i
    return headers
            
def loadLabels(fn, headers, labels, labelpath):
    with open('{0}/data/{1}.txt'.format(labelpath, fn), 'r') as data:
        line = data.readline()
        vals = line.split(' ')
        for key in labels:
            labels[key].append(float(vals[headers[key]]))
    
def compFlow(prev, cur, **options):
    gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    flow, avgflow = getflow(prevgray, gray, **options)
    
    cplx = avgflow[:,:,0] + avgflow[:,:,1] * 1j
    cplx = cplx.flatten()
    mag = np.absolute(cplx)
    ang = np.angle(cplx)
    return mag.tolist() + ang.tolist()

def predSpeed(im, prev, cur, icmp, labels, **options):
    if prev is None:
        return im, icmp
    parampath = '.'
    if 'parampath' in options:
        parampath = options['parampath']

    with open('{0}/parameters.txt'.format(parampath), 'r') as paramfile:
        rseg, cseg = paramfile.readline().split(',')
        rseg = int(rseg)
        cseg = int(cseg)
        coef = paramfile.readline().split(',')
        coef = np.array(map(float, coef))
    flow = compFlow(prev, cur, rseg=rseg, cseg=cseg)
    regr = linear_model.LinearRegression()
    regr.coef_ = coef
    regr.intercept_ = True 
    gtspeed = labels['vf'][-1]
    speed = regr.predict([flow])[0]

    text = 'Predicted Speed: {:02.2f}km/h. Ground Truth: {:02.2f}km/h'.format(speed, gtspeed)
    h = icmp.shape[0]
    coord = (20, h*1/4)
    fontface = cv2.FONT_HERSHEY_SIMPLEX;
    icmp = cv2.putText(img=icmp, text=text, org=coord, fontFace=fontface, 
            fontScale=0.6, color=bgr('k'), thickness=2, lineType=8);
    return im, icmp

def trainSpeed(flows, labels, rseg, cseg, **options):
    pctTrain = 0.8
    parampath = '.'
    if 'parampath' in options:
        parampath = options['parampath']

    numTest = int(round(len(flows)*(1-pctTrain)))
    # Split the data into training/testing sets
    X_train = flows[:-numTest]
    X_test = flows[-numTest:]
    
    # Split the targets into training/testing sets
    y_train = labels['vf'][:-numTest]
    y_test = labels['vf'][-numTest:]
    
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=True)
    # Train the model using the training sets
    regr.fit(X_train, y_train)
    
    # write coefficients into a file
    with open('{0}/parameters.txt'.format(parampath), 'w') as paramfile:
        paramfile.write(','.join(map(str, [rseg, cseg])) + '\n')
        paramfile.write(','.join(map(str, regr.coef_)))

    # The coefficients
    # print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))
    
# def main():

# if __name__ == "__main__":
    # main()
