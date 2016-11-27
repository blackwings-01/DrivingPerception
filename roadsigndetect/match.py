from os import listdir
from os.path import isfile, join, splitext
import numpy as np
import cv2
from matplotlib import pyplot as plt
from colors import *
from path import *

# Match road sign in image 

signs = {}
for f in listdir(SIGN_PATH):
    fn, ext = splitext(f)
    if isfile(join(SIGN_PATH, f)) and ext=='.png':
        signs[fn] = join(SIGN_PATH, f)

def match(img1, img2, **options):
    if 'draw' in options:
        draw = options['draw']
    else:
        draw = False 
    if 'matchColor' in options:
        matchColor = options['matchColor']
    else:
        matchColor = 'g' 
    if 'singlePointColor' in options:
        singlePointColor = options['singlePointColor']
    else:
        singlePointColor = 'b' 

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    
    if draw:
        draw_params = dict(matchColor = bgr(matchColor),
                           singlePointColor = bgr(singlePointColor),
                           matchesMask = matchesMask,
                           flags = 0)
        
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        
        plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)),plt.show()


def main():
    # img1 = cv2.imread(signs['stop_sign'])
    # img2 = cv2.imread('{0}{1}'.format(DATA_PATH,
        # 'NRM_20160615005414_Goluk_T1_800865_png/NRM_20160615005414_Goluk_T1_800865_552.png'))
    img1 = cv2.imread(signs['yield'])
    img2 = cv2.imread('{0}{1}'.format(DATA_PATH,
        'NRM_20160615005414_Goluk_T1_800865_png/NRM_20160615005414_Goluk_T1_800865_483.png'))
    match(img1, img2, draw=True)

if __name__ == "__main__":
    main()
