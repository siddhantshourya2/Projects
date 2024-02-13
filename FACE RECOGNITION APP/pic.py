import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

    imgpath = 'surprize.jpg'
    image = cv2.imread(imgpath)
    analyze = DeepFace.analyze(image,actions=['emotion'])
    print(analyze)
    print(analyze[0]['dominant_emotion'])
    break

cap.release()
cv2.destroyAllWindows()
