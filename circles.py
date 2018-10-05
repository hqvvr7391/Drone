from picamera.array import PiRGBArray
from picamera import PiCamera

import numpy as np
import time
import cv2

file = 'Video2.avi'
camera = cv2.VideoCapture(file)

time.sleep(0.1)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_thres, high_thres):
    return cv2.Canny(img, low_thres, high_thres)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def hsv_split(img, lowthres, highthres):
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, lowthres, lowthres])
    upper = np.array([100, highthres, highthres])

    mask = cv2.inRange(imghsv, lower, upper) 

    return mask
     

def split(img, thres):
    b, g, r = cv2.split(img)

    b = gaussian_blur(b, 3)
    g = gaussian_blur(g, 3)
    r = gaussian_blur(r, 3)
    ret, bT = cv2.threshold(b, thres, 255, 0)
    ret, gT = cv2.threshold(g, thres, 255, 0)
    ret, rT = cv2.threshold(r, thres, 255, 0)

    thres = cv2.bitwise_or(bT, gT)
    thres = cv2.bitwise_or(thres, rT)

    return thres


def draw_lines(img, lines, color=[0,0,255], thickness=2):
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_circles(img, circles, color=[255, 255, 0], thickness=2):
    if circles is not None:
        for circle in circles[0, :]:
            cv2.circle(img, (circle[0], circle[1]), circle[2], color, thickness)


def draw_fitline(img, lines, color=[255, 0, 0], thickness=5):
    return 0


def hough_lines(img, rho, theta, thres, min_len, max_gap):
    lines = cv2.HoughLinesP(img, rho, theta, thres, np.array([]), minLineLength=min_len, maxLineGap=max_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img


def hough_circles(img, a, par1, par2, minR, maxR):
    return cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, a, param1 =par1, param2 = par2, minRadius = minR, maxRadius = maxR)


def find_contours(img):

    cimg, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img, contours, min_size, max_size):
    circles = list()
    cnt = 0

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= min_size:
            if area <= max_size:
                ellipse = cv2.fitEllipse(contours[i])
         
                cv2.ellipse(img, ellipse, (255, 255, 0), 2)

                circles.append(ellipse)
                
    return circles


def diagonal_sort(img, circles):
    sort = list(circles)

    m = img.shape[0] / img.shape[1]
    b1 = 0
    
    mr = -1 / m

    for circle in sort:
        b2 = circle[0][1] - mr * circle[0][0]
        p = (b2 - b1)
        p = (int(p / (m - mr)), int(m * p/(m - mr) + b1))
        
        la = list(circle)
        la.append(p)
        sort[sort.index(circle)] = tuple(la)
        

    sort = sorted(sort, key=lambda p: p[3][0])
    
    
    return sort


def draw_path(img, circles, color=[0, 255, 255], thickness=5):
    
    sort = diagonal_sort(img, circles)

    if len(sort) >= 2:
   
        for i in range(len(sort)-1):
           
            #cv2.putText(img, 'a', \
            #        (int(circles[i][0][0]), int(circles[i][0][1])),\
            #        cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)


            #cv2.line(img, (int(circles[i][0][0]), int(circles[i][0][1])), \
            #        (int(circles[i+1][0][0]), int(circles[i+1][0][1])), \
            #        color, thickness)           
            cv2.line(img, (int(sort[i][0][0]), int(sort[i][0][1])), \
                    (int(sort[i+1][0][0]), int(sort[i+1][0][1])), \
                     color, thickness)


def weight_img(img, initial_img, a=1, b=1, r=0):
    return cv2.addWeighted(initial_img, a, img, b, r)




while (camera.isOpened()):

    start = time.time()

    ret, frame = camera.read()
    frame = cv2.resize(frame, (640, 480))

    image = frame.copy()
    img2 = image.copy()

    imgray = grayscale(img2)
    g_img = gaussian_blur(img2, 3)

    split_img = split(g_img, 150)
    
    canny_img = canny(split_img, 70, 140)


    contours = find_contours(split_img)
    #print(contours, '\n')
    circles = draw_contours(image, contours, 60, 3200)

    ###print(circles, end='\n')
    
    #hough_img = hough_lines(canny_img, 1, 1 *np.pi/360, 30, 60 ,10)

    ###hough_circle = hough_circles(canny_img, 10, 40, 25, 0, 100)
    ###draw_circles(image, hough_circle)

    sort = diagonal_sort(image, circles)
 
    draw_path(image, sort)
           
    #result = weight_img(hough_img, image)

    cv2.imshow('result',image)

    ###cv2.imshow('sdf', split_img)

    print(time.time() - start, end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
