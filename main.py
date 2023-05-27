import cv2
import numpy as np
nPlateCascade= cv2.CascadeClassifier("haarcascades/haarcascade_russian_plate_number.xml")
#frame width
fw=648
#frame height
fh=488
cap=cv2.VideoCapture(0)
cap.set(3,fw)
cap.set(4,fh)
cap.set(10,150)
while True:
    success,img=cap.read()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    numberPlates=nPlateCascade.detectMultiScale(imgGray,1,1,4)
    for(x,y,w,h) in numberPlates:
        area=w*h
        if area > 500:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),255)
            imgRoi= img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("result",img)
    if cv2.waitKey(1) & 0xFF ==ord('s'):
        cv2.imwrite("")

