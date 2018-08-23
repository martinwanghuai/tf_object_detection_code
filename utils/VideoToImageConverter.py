'''
Created on 19 Aug 2018

@author: martinwang
'''
import cv2

vc = cv2.VideoCapture('/Users/martinwang/eclipse-workspace/tf_object_detection_code/utils/car.avi')
c=0
rval = vc.isOpened()

while rval:
    c = c + 1
    rval, frame = vc.read()
    
    if rval:
        cv2.imwrite('/Users/martinwang/eclipse-workspace/tf_object_detection_code/utils/images/' + str(c) + '.jpg', frame)
        print("handle:" + str(c) + ".jpg")
        cv2.waitKey(1)
    else:
        break;
vc.release();