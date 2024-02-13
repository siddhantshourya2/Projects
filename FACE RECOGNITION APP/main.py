import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotion = result[0]["dominant_emotion"]
    
    txt = str(emotion)

    cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
#imgpath = 'surprize.jpg'
#image = cv2.imread(imgpath)
#analyze = DeepFace.analyze(image,actions=['emotion'])
#print(analyze)
#print(analyze[0]['dominant_emotion'])
#break

cap.release()
cv2.destroyAllWindows()
