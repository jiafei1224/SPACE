import cv2
import numpy as np
import os

def color(path,savedpath):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv ,11)

    #Green color
    lowerG = np.array([41, 153, 0])
    upperG = np.array([63, 255, 255])

    maskG = cv2.inRange(blur, lowerG, upperG)
    res = cv2.bitwise_and(image,image, mask= maskG)
    
    valuegreen=np.sum(maskG)
#     print(valuegreen)


    is_all_zero = np.all((maskG == 0))
#     checker4=0
    if is_all_zero or valuegreen < 4500:
#         print('Array contains only 0')
        checker=0
    else:
#         print('Yellow')
        checker=1

    if checker==1:    
    #     cv2.imshow("mask ",maskY)
        cv2.imwrite(savedpath+'/Green.jpg', maskG)
    ############################################################################
    #bLUE color
    lowerY = np.array([90, 0, 0])
    upperY = np.array([140, 255, 255])

    maskY = cv2.inRange(blur, lowerY, upperY)
    res = cv2.bitwise_and(image,image, mask= maskY)            


    is_all_zero = np.all((maskY == 0))
#     checker2=0
    if is_all_zero:
#         print('Array contains only 0')
        checker2=0
    else:
#         print('Blue')
        checker2=1

    if checker2==1:    
    #     cv2.imshow("mask ",maskY)
    
        cv2.imwrite(savedpath+'/Blue.jpg', maskY)

        ############################################################################
    #Red color
    lowerR = np.array([0,50,50])
    upperR = np.array([10,255,255])

    maskR = cv2.inRange(blur, lowerR, upperR)
    res = cv2.bitwise_and(image,image, mask= maskR)            


    is_all_zero = np.all((maskR == 0))
#     checker3=0
    if is_all_zero:
#         print('Array contains only 0')
        checker3=0
    else:
#         print('Red')
        checker3=1

    if checker3==1:    
    #     cv2.imshow("mask ",maskY)
        cv2.imwrite(savedpath+'/Red.jpg', maskR)
#     print(checker3)
    # cv2.imshow('stack', np.hstack([image, res]))
    # cv2.waitKey(0)
        ############################################################################
    #YELLOW color
    lowerB = np.array([21, 39, 64])
    upperB = np.array([40, 255, 255])

    maskB = cv2.inRange(blur, lowerB, upperB)
    res = cv2.bitwise_and(image,image, mask= maskB)            


    is_all_zero = np.all((maskB == 0))
#     checker4=0
    if is_all_zero:
#         print('Array contains only 0')
        checker4=0
    else:
#         print('Yellow')
        checker4=1

    if checker4==1:    
    #     cv2.imshow("mask ",maskY)
        cv2.imwrite(savedpath+'/Yellow.jpg', maskB)
        
   
    return checker,checker2,checker3,checker4
    # cv2.imshow('stack', np.hstack([image, res]))
    # cv2.waitKey(0)
    
def color2(path,savedpath,chk1,chk2,chk3,chk4):
    image = cv2.imread(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.medianBlur(hsv ,11)

    #Green color
    lowerG = np.array([41, 153, 0])
    upperG = np.array([63, 255, 255])

    maskG = cv2.inRange(blur, lowerG, upperG)
    res = cv2.bitwise_and(image,image, mask= maskG)            




    if chk1==1:
        
    #     cv2.imshow("mask ",maskG)
        cv2.imwrite(savedpath+'/Green.jpg', maskG)
    ############################################################################
    #bLUE color
    lowerY = np.array([90, 0, 0])
    upperY = np.array([140, 255, 255])

    maskY = cv2.inRange(blur, lowerY, upperY)
    res = cv2.bitwise_and(image,image, mask= maskY)            


    if chk2==1:   
    #     cv2.imshow("mask ",maskY)
    
        cv2.imwrite(savedpath+'/Blue.jpg', maskY)

        ############################################################################
    #Red color
    lowerR = np.array([0,50,50])
    upperR = np.array([10,255,255])

    maskR = cv2.inRange(blur, lowerR, upperR)
    res = cv2.bitwise_and(image,image, mask= maskR)            


    if chk3==1:    
    #     cv2.imshow("mask ",maskY)
        cv2.imwrite(savedpath+'/Red.jpg', maskR)
#     print(checker3)
    # cv2.imshow('stack', np.hstack([image, res]))
    # cv2.waitKey(0)
        ############################################################################
    #YELLOW color
    lowerB = np.array([21, 39, 64])
    upperB = np.array([40, 255, 255])

    maskB = cv2.inRange(blur, lowerB, upperB)
    res = cv2.bitwise_and(image,image, mask= maskB)            


    if chk4==1:    
    #     cv2.imshow("mask ",maskY)
        cv2.imwrite(savedpath+'/Yellow.jpg', maskB)
    
    # cv2.imshow('stack', np.hstack([image, res]))
    # cv2.waitKey(0)

for fileno in range(1,11):
    filepath= 'C:/Users/duanj/Desktop/testingtensor/saved masks/'+str(fileno)
    os.mkdir(filepath)
    print("Fileno:"+str(fileno))
    for frame in range(1,151):
        path ='C:/Users/duanj/Desktop/testingtensor/test frame/'+str(fileno)+'/'+str(frame)+'.png'
        savedpath= str(filepath)+'/'+str(frame)
        os.mkdir(savedpath)
    
        if frame==1:
            color(path,savedpath)
            chk1=color(path,savedpath)[0]
            chk2=color(path,savedpath)[1]
            chk3=color(path,savedpath)[2]
            chk4=color(path,savedpath)[3]
            print(color(path,savedpath)[0])
            print(color(path,savedpath)[1])
            print(color(path,savedpath)[2])
            print(color(path,savedpath)[3])
        else:
            color2(path,savedpath,chk1,chk2,chk3,chk4)