import cv2
import numpy as np
import face_recognition
import os

path ='ImagesATT'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curimg =cv2.imread(f'{path}/{cls}')
    images.append(curimg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findencode(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencode(images)
print(len(encodelistknown))

cap = cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgsmall = cv2.resize(img, (0,0),None,0.25,0.25)
    imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)

    facecurLoc = face_recognition.face_locations(imgsmall)
    encodecur = face_recognition.face_encodings(imgsmall,facecurLoc)

    for encodeface,faceloc in zip(encodecur,facecurLoc):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedist=face_recognition.face_distance(encodelistknown,encodeface)
        print(facedist)
        matchindex=np.argmin(facedist)

        if matches[matchindex]:
            name=classNames[matchindex].upper()
            print(name)
            x1,y1,x2,y2=faceloc
            x1, y1, x2, y2=x1*4,y1*4,x2*4,y2*4
            cv2.rectangle(img,(y2,x1),(y1,x2),(0,255,0),2)
            cv2.rectangle(img,(y2,x2-35),(y1,x2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(y2+6,x2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)





    cv2.imshow('webcam',img)
    key = cv2.waitKey(1)
    if key == 27:
        break
