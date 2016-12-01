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

def loadMatch(frame, fn, matches):
    if fn not in matches:
        return frame
    for sn in matches[fn]:
        mc = matches[fn][sn]
        if (len(mc)==0):
            continue
        # cnrs = mc['cnrs']
        # frame = cv2.polylines(img=frame, pts=[cnrs], isClosed=True, color=bgr('b'),
                # thickness=3, lineType=cv2.LINE_AA)
        ctr = mc['ctr'] 
        frame = cv2.circle(img=frame, center=ctr, radius=2, color=bgr('r'), thickness=-1,
                    lineType=cv2.LINE_AA)
        frame = drawLabel(img=frame, label=sn, coord=ctr)
    return frame

def roadSignMatching(frame):
    sign = cv2.imread(signs['keep_right'])
    sign = cv2.GaussianBlur(sign,(5,5),0)
    img = match(sign, frame, draw=True, drawKeyPoint=False, ratioTestPct=0.7, minMatchCnt=5)
    return img

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
    (opts, args) = parser.parse_known_args()

    files = [f for f in listdir(opts.path) if isfile(join(opts.path, f)) and f.endswith('.png')]
    files = sorted(files)

    if opts.mode in ['loadmatch', 'all']:
        matches = mcread(opts.path)

    img = None
    org = None
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

        if opts.mode == 'roadsign':
            im = roadSignMatching(im) 
        elif opts.mode == 'loadmatch':
            im = loadMatch(im, fn, matches) 
        elif opts.mode == 'detlight':
            im,org = detlight(im, mode='compare') 
        elif opts.mode == 'all':
            im,_ = detlight(im, mode='label') 
            im = loadMatch(im, fn, matches) 

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if org is not None:
            org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

        if img is None:
            if org is not None:
                plt.subplot(2,1,1)
                imgo = plt.imshow(org)
                plt.subplot(2,1,2)
                img = plt.imshow(im)
            else:
                img = plt.imshow(im)
        else:
            if org is not None:
                imgo.set_data(org)
                img.set_data(im)
            else:
                img.set_data(im)
        plt.pause(opts.delay)
        plt.draw()

if __name__ == "__main__":
    main()
