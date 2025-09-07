import cv2 as cv
import os
import numpy as np

def reshape(img, scale = 0.6):
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] * scale)
    dimensions = (width , height)
    
    return cv.resize(img, dimensions , interpolation=cv.INTER_AREA)


haar_cascade = cv.CascadeClassifier('haar_face.xml')


persons = []
for name in os.listdir(r"input_ the_path_Directory, _where_image_folders_are_stored.."):
    persons.append(name)
    
# print(persons)

DIR = r"input_ the_path_Directory, _where_image_folders_are_stored.."

features =[]
names = []

def train():
    for name in persons:
        path = os.path.join(DIR, name)
        label = persons.index(name)
        
        for img in os.listdir(path):
            img_path= os.path.join(path,img)
            
            img = cv.imread(img_path)
            # reshape(img)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faceFound = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1,minNeighbors=15)
            #generalli minimum minNeighbors value ives good results

            for(x,y,w,h)  in faceFound:
                faces_at_region = gray_img[ y:y+h, x:x+w]
                features.append(faces_at_region)
                names.append(label)
                
train()

# print(len(features))
# print(persons)  
# print(names)
            
        
face_recog = cv.face.LBPHFaceRecognizer_create()
face_recog.train(np.array(features, dtype='object'),np.array(names))

inp =cv.imread(r"input_ the_path_person_to_be_recognized")
inp  = reshape(inp)
cv.imshow("the input given " , inp)

gray_inp = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)


draw_rect = haar_cascade.detectMultiScale(gray_inp, 1.1, 3)

if len(draw_rect) > 0:
    # Pick the largest face (max area)
    (x, y, w, h) = max(draw_rect, key=lambda f: f[2] * f[3])

    faces_at_region = gray_inp[ y:y+h, x:x+w]
    label, confidence = face_recog.predict(faces_at_region)
    print(f'Label {persons[label]} is the name of the person')
    
    cv.putText(inp , str(persons[label]) , (20,20) , cv.FONT_HERSHEY_COMPLEX , 1.0, (0,255,0) , 2)
    
    cv.rectangle(inp, (x,y) , (x+w ,y+h) , (0,255,0)  ,2)
    
cv.imshow("the  detected face" , inp)
cv.waitKey(0)
