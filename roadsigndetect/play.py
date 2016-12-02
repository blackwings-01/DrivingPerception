from os import listdir
from os.path import isfile, join, splitext
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from path import *
from match import *
from util import * 
from speeddet import *
from lightdet import *

def roadSignMatching(frame, org):
    sign = cv2.imread(signs['keep_right'])
    sign = cv2.GaussianBlur(sign,(5,5),0)
    img = match(sign, frame, org, draw=True, drawKeyPoint=False, ratioTestPct=0.7, minMatchCnt=5)
    return img

def play(opts):
    files = [f for f in listdir(opts.path) if isfile(join(opts.path, f)) and f.endswith('.png')]
    files = sorted(files)

    if opts.mode in ['loadmatch', 'all']:
        matches = mcread(opts.path)
    if opts.mode in ['trainspeed', 'all']:
        headers = loadHeader('{0}/../oxts'.format(opts.path))
        labels = dict(vf=[], wf=[])
        flows = []

    img = None
    icmp = None
    porg = None
    plt.figure(dpi=140)
    for i, impath in enumerate(files): 
        fn, ext = splitext(impath)
        if i<opts.startframe:
            continue
        if opts.endframe>0 and i>opts.endframe:
            break
        if opts.numframe>0 and i>(opts.startframe + opts.numframe):
            break

        root, ext = splitext(impath)
        im = cv2.imread(join(opts.path, impath), cv2.IMREAD_COLOR)
        org = im.copy()

        if opts.mode == 'roadsign':
            im = roadSignMatching(im, org) 
        elif opts.mode == 'loadmatch':
            im,_ = loadMatch(im, org, icmp, fn, matches) 
        elif opts.mode == 'detlight':
            im,icmp,_ = detlight(im, org, mode='compare') 
        elif opts.mode == 'flow':
            if porg is not None:
                im = detflow(im, porg, org, flowmode='avgflow')
        elif opts.mode == 'trainspeed':
            if porg is not None:
                flow = compFlow(porg, org, rseg=opts.rseg, cseg=opts.cseg)
                flows.append(flow)
                loadLabels(fn, headers, labels, '{0}/../oxts'.format(opts.path))
        elif opts.mode == 'all':
            h,w,_ = im.shape
            h = 200
            icmp = np.ones((h,w,3), np.uint8) * 255
            im, icmp = predSpeed(im, porg, org, icmp, labels, rseg=opts.rseg, cseg=opts.cseg)
            im, icmp = detlight(im, org, mode='label', icmp=icmp) 
            im, icmp = loadMatch(im, org, icmp, fn, matches) 
            loadLabels(fn, headers, labels, '{0}/../oxts'.format(opts.path))
        porg = org.copy()

        if opts.mode in ['trainspeed']:
            continue
        
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if icmp is not None:
            icmp = cv2.cvtColor(icmp, cv2.COLOR_BGR2RGB)

        if img is None:
            if icmp is not None:
                plt.subplot(2,1,1)
                img = plt.imshow(im)
                plt.subplot(2,1,2)
                imgo = plt.imshow(icmp)
            else:
                img = plt.imshow(im)
        else:
            if icmp is not None:
                imgo.set_data(icmp)
                img.set_data(im)
            else:
                img.set_data(im)
        plt.pause(opts.delay)
        plt.draw()

    if (opts.mode=='trainspeed'):
        trainSpeed(flows, labels, opts.rseg, opts.cseg)

def main():
    usage = "Usage: play [options --path]"
    parser = argparse.ArgumentParser(description='Visualize a sequence of images as video')
    parser.add_argument('--path', dest='path', action='store', 
            default='{0}2011_09_26-3/data'.format(KITTI_PATH),
            help='Specify path for the image files')
    parser.add_argument('--delay', dest='delay', nargs='?', default=0.05, type=float,
            help='Amount of delay between images')
    parser.add_argument('--start-frame', dest='startframe', nargs='?', default=0, type=int,
            help='Starting frame to play')
    parser.add_argument('--end-frame', dest='endframe', nargs='?', default=-1, type=int,
            help='Ending frame to play, -1 for last frame')
    parser.add_argument('--num-frame', dest='numframe', nargs='?', default=-1, type=int,
            help='Number of frame to play, -1 for all frames')
    parser.add_argument('--mode', dest='mode', action='store', default='roadsign')
    parser.add_argument('--rseg', dest='rseg', nargs='?', default=3, type=int,
            help='Number of vertical segmentation in computing averaged flow')
    parser.add_argument('--cseg', dest='cseg', nargs='?', default=4, type=int,
            help='Number of horizontal segmentation in computing averaged flow')
    (opts, args) = parser.parse_known_args()
    play(opts)

if __name__ == "__main__":
    main()
