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
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image=gray,threshold1=50,threshold2=200,apertureSize=3)
    plt.imshow(edges, cmap=plt.cm.binary)
    plt.show()
    # cv2.imshow('image', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    minLineLength = 30
    maxLineGap = 5
    global lines
    lines = cv2.HoughLinesP(edges,rho=10,theta=np.pi/180*10,threshold=80,
                minLineLength=minLineLength,maxLineGap=maxLineGap)
    global line
    for line in lines:
        x1,y1,x2,y2 = line[0]
        img = cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    # cv2.imwrite('houghlines5.jpg',img)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()
