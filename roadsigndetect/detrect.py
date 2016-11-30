import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Detect rectangle in image

def main():
    usage = "Usage: detrect [options --file]"
    parser = argparse.ArgumentParser(description='Detect rectangles in an image')
    parser.add_argument('--file', dest='filepath', action='store', 
            default='../kitti/2011_09_26/2011_09_26_drive_0048_sync/image_03/data/0000000000.png',
            help='specify path for the image file')

    (opts, args) = parser.parse_known_args()

    global img
    img = cv2.imread(opts.filepath, cv2.IMREAD_COLOR)

    # contour detection
    # img = cv2.blur(img,(3,3));
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(gray,127,255,0)
    # image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # lcontours = []
    # for c in contours:
        # peri = cv2.arcLength(c, True)
        # if peri>50 and peri < 200:
            # lcontours.append(c)
        # approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # plt.ion()
    # for i, c in enumerate(lcontours):
        # img = cv2.drawContours(img, lcontours, contourIdx=i, color=(0,255,0), thickness=1)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # _ = raw_input("Press any key to continue")
        # plt.show()
        # plt.close()
    # return

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray,threshold1=50,threshold2=200,apertureSize=3)
    plt.imshow(edges, cmap=plt.cm.binary)
    plt.show()
    return

    minLineLength = 30
    maxLineGap = 5
    global lines
    lines = cv2.HoughLinesP(edges,rho=10,theta=np.pi/180*10,threshold=80,
                minLineLength=minLineLength,maxLineGap=maxLineGap)
    global line
    for line in lines:
        x1,y1,x2,y2 = line[0]
        img = cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()
