import cv2
import os
import time

algo = "HaarCascade Algorithm.xml"
haar = cv2.CascadeClassifier(algo)  # Initializing Algorithm

cam = cv2.VideoCapture(0)    # Reading Camera feed

# Creating database:
user = input("Enter your Name : ")
# Change this path variable or take is as dynamic
path = 'E:\\PRANK...@@\\Dark Code [Illegal]\\AI MASTERCLASS\\Image Processing\\Face Detection\\Database\\'+user
if not os.path.isdir(path):
    os.mkdir(path)

count = 1 
while count<31:
    _, img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = haar.detectMultiScale(grayImg, 1.4 , 4) # deploying Algo  
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        onlyFace = grayImg[y:y+h, x:x+w]     # croping ony Face through slicing the pixel.
        resizeImg = cv2.resize(onlyFace, (130, 100))    # Resize using cv2
        cv2.imwrite("%s/%s.jpg" %(path, count), resizeImg)            
        count += 1
        #time.sleep(1)
        
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(1) & 0xFF
    if key==ord('x'):  # use key==27 for 'Esc' key
        break

cam.release()
cv2.destroyAllWindows()
