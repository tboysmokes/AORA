import cv2 as cv
import numpy as np

dataSet = {
    "tumi": [
        "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.07__1_-removebg-preview.png", "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.07__3_-removebg-preview.png", "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.07-removebg-preview.png",
          "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.09-removebg-preview.png", "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.09__1_-removebg-preview.png", "images/tumi/WhatsApp_Image_2024-06-14_at_11.06.08__2_-removebg-preview.png"
    ],
    "tomiwa": [
        "images/tomiwa/IMG_0216 11.20.51 AM.jpg", "images/tomiwa/IMG_0269.jpg", "images/tomiwa/IMG_2455-removebg-preview.png", "images/tomiwa/IMG_2456-removebg-preview.png",
        "images/tomiwa/IMG_2457-removebg-preview.png", "images/tomiwa/IMG_2458-removebg-preview.png", "images/tomiwa/IMG_2459-removebg-preview.png", "images/tomiwa/IMG_2460-removebg-preview.png",
        "images/tomiwa/IMG_2546.JPG", "images/tomiwa/IMG_2547.JPG", "images/tomiwa/IMG_2550.JPG", "images/tomiwa/IMG_2553.jpg", "images/tomiwa/IMG_2554.jpg"
    ],
    "sean": [
        "images/seam/IMG_1052.jpg", "images/seam/IMG_5567.png", "images/seam/IMG_5568.png", "images/seam/IMG_5569.png", "images/seam/sean1-removebg-preview.png", "images/seam/sean1.jpeg",
        "images/seam/sean2-removebg-preview.png", "images/seam/sean3-removebg-preview.png", "images/seam/sean3.jpeg", "images/seam/sean4-removebg-preview.png", "images/seam/IMG_2555.jpg", "images/seam/IMG_2556.jpg"
    ]
}

labels = []
features= []

image_size = (100, 100)
label_dict = {name: i for i, name in enumerate(dataSet.keys())}
label_dict_finder = []

for i, name in enumerate(dataSet.keys()):
    label_dict_finder.append({'name': name, 'id': i})


for name, pictures in dataSet.items():
    for picture in pictures:
        img = cv.imread(picture)
        if img is None:
            continue
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resize_grey = cv.resize(grey, image_size)
        features.append(resize_grey)
        labels.append(label_dict[name])

labels = np.array(labels)
print(labels)
features = np.array(features)


face_recognizer = cv.face.LBPHFaceRecognizer.create()
face_recognizer.train(features, labels=labels)

video_capture = cv.VideoCapture(0)

def get_name(label):
    for data in label_dict_finder:
        if label == data['id']:
            return data['name']
    return 'unkown'


faceDetector = True
while faceDetector:
    ret, frame = video_capture.read()
    if not ret:
        break

    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_cascacde = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces= face_cascacde.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for(x, y, w, h) in faces:
        face_region = grey_frame[y:y+h, x:x+w]
        resize_face_region = cv.resize(face_region, image_size)

        label, confidence = face_recognizer.predict(resize_face_region)
        print(label)
        for data in label_dict_finder:
            if label:
                if label == data['id']:
                    name = data['name']
                else: 
                    name = 'unknow'
                    print('unknow')
            else: print('no predicted label')


        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(frame, name, (x, y-10), cv.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), 2)

    cv.imshow('monitor', frame)
    cv.waitKey(2)


# python3 computerVision.py