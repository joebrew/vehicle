import cv2
import numpy as np
import os
os.chdir('/home/joebrew/Documents/vehicle')

class Kordinat:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class Sensor:
    def __init__(self,kordinat1,kordinat2,frame_weight,frame_lenght):
        self.kordinat1=kordinat1
        self.kordinat2=kordinat2
        self.frame_weight=frame_weight
        self.frame_lenght =frame_lenght
        self.mask=np.zeros((frame_weight,frame_lenght,1),np.uint8)*abs(self.kordinat2.y-self.kordinat1.y)
        self.full_mask_area=abs(self.kordinat2.x-self.kordinat1.x)
        cv2.rectangle(self.mask,(self.kordinat1.x,self.kordinat1.y),(self.kordinat2.x,self.kordinat2.y),(255),thickness=cv2.FILLED)
        self.stuation=False
        self.car_number_detected=0


# video=cv2.VideoCapture("video1.mp4")
video=cv2.VideoCapture("x.mp4")

ret,frame=video.read()
# vals = [0, 450, 0, 450]
vals = [300,550,380,800]
# cropped_image= frame[0:450, 0:450]
cropped_image = frame[vals[0]:vals[1], vals[2]:vals[3]]
fgbg=cv2.createBackgroundSubtractorMOG2()
# Sensor1 = Sensor(
#     Kordinat(1, cropped_image.shape[1] - 35),
#     Kordinat(340, cropped_image.shape[1] - 30),
#     cropped_image.shape[0],
#     cropped_image.shape[1])
Sensor1 = Sensor(
    Kordinat(50, 90),
    Kordinat(150, 90),
    cropped_image.shape[0],
    cropped_image.shape[1]
)

kernel=np.ones((5,5),np.uint8)
font=cv2.FONT_HERSHEY_TRIPLEX

# directory to save the ouput frames
pathIn = "frames/"

counter = 0
last_car = 0
while (1):
    counter = counter + 1
    # print(str(counter))
    ret,frame=video.read()
    # resize frame
    cropped_image= frame[vals[0]:vals[1], vals[2]:vals[3]]
    # make morphology for frame
    deleted_background=fgbg.apply(cropped_image)
    opening_image=cv2.morphologyEx(deleted_background,cv2.MORPH_OPEN,kernel)
    ret,opening_image=cv2.threshold(opening_image,125,255,cv2.THRESH_BINARY)

    # detect moving anything
    cnts,_=cv2.findContours(opening_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    result=cropped_image.copy()

    zeros_image=np.zeros((cropped_image.shape[0], cropped_image.shape[1], 1), np.uint8)

    # detect moving anything with loop
    minus = 40
    for cnt in cnts:
        x,y,w,h=cv2.boundingRect(cnt)
        if (w>15 and h>15  and w<160 and h<160 ): # if (w>75 and h>75  and w<160 and h<160 ):
            cv2.rectangle(result,(x,y-minus),(x+w,y+h-minus),(255,0,0),thickness=2)
            cv2.rectangle(zeros_image,(x,y-minus),(x+w,y+h-minus),(255),thickness=cv2.FILLED)

    # detect whether there is car via bitwise_and
    mask1=np.zeros((zeros_image.shape[0],zeros_image.shape[1],1),np.uint8)
    mask_result=cv2.bitwise_or(zeros_image,zeros_image,mask=Sensor1.mask)
    white_cell_number=np.sum(mask_result==255)

    # detect to control whether car is passing under the red line sensor
    sensor_rate=white_cell_number/Sensor1.full_mask_area
    if sensor_rate>0:
        print("result : ",sensor_rate)
    # if car is passing under the red line sensor . red line sensor is yellow color.
    # define sensor threshold
    sensor_threshold = 0.1
    sensor_max = 0.8
    time_since_last_car = counter - last_car
    if (sensor_rate>=sensor_threshold and  sensor_rate< sensor_max and Sensor1.stuation==False and time_since_last_car > 10):
        print('Time since last car: ' + str(time_since_last_car))
        last_car = counter
        # draw the red line
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0,255, 0,), thickness=cv2.FILLED)
        Sensor1.stuation = True
    elif (sensor_rate<sensor_threshold and Sensor1.stuation==True) :
        # draw the blue line
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0, 0,255), thickness=cv2.FILLED)
        Sensor1.stuation = False
        Sensor1.car_number_detected+=1
    else :
        # draw the red line
        cv2.rectangle(result, (Sensor1.kordinat1.x, Sensor1.kordinat1.y), (Sensor1.kordinat2.x, Sensor1.kordinat2.y),
                      (0, 0, 255), thickness=cv2.FILLED)



    cv2.putText(result,str(Sensor1.car_number_detected),(Sensor1.kordinat1.x,150),font,2,(255,255,255))


    cv2.imshow("video", result)
    if counter % 1 == 0:
        cv2.imwrite(pathIn + str(counter) + '.png', result)

    #cv2.imshow("mask_result", mask_result)
    #cv2.imshow("zeros_image", zeros_image)
    #cv2.imshow("opening_image", opening_image)

    k=cv2.waitKey(30) & 0xff
    if k == 27 :
        break

video.release()
cv2.destroyAllWindows()


# specify video name
pathOut = 'vehicle_detection_v3.mp4'

# specify frames per second
fps = 60.0

import os
import re
frame_array = []
files = [f for f in os.listdir(pathIn) if os.path.isfile((pathIn + f))]

files.sort(key=lambda f: int(re.sub('\D', '', f)))

for i in range(len(files)):
    filename = pathIn + files[i]

    # read frames
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    frame_array.append(img)

# Finally, we will use the below code to make the object detection video:

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()