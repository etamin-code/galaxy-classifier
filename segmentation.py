'''this module work with segmentation of the images'''
import cv2
import numpy as np
from copy import copy

def viewImage(image):
    '''show image'''
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmantate(gray, step=15):
    '''return segmantated image
    step - step between minimal differences of 
    brights of pixels'''
    high = 255
    segmantated = copy(gray)
    while(1):  
        low = high - step
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(segmantated, col_to_be_changed_low, col_to_be_changed_high)
        segmantated[curr_mask > 0] = (high)    
        high -= step
        if(low <= step): # make almost black pixels completely black
            curr_mask = cv2.inRange(segmantated, np.array([0]), col_to_be_changed_high)
            segmantated[curr_mask > 0] = (0)
            break
    return segmantated

def border(gray):
    '''return border of the object on the image as other image'''
    high = 165
    step = 30
    border = copy(gray)
    curr_mask = cv2.inRange(border, np.array([165]), np.array([255]))
    border[curr_mask > 0] = (0)
    curr_mask = cv2.inRange(border, np.array([0]), np.array([105]))
    border[curr_mask > 0] = (0)
    for i in range(2):  
        low = high - step
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(border, col_to_be_changed_low, col_to_be_changed_high)
        border[curr_mask > 0] = (high)    
        high -= step
    return border


if __name__ == '__main__':
    image = cv2.imread(r'D:\programing\second_semester\home_work\galaxy-zoo-the-galaxy-challenge\images_training_rev1\images_training_rev1\135093.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seg = segmantate(gray)
    viewImage(seg)
    bor = border(gray)
    viewImage(bor)


