#Reconoce el rostro usando la cámara de la laptop
import cv2
import utils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

GREEN = (0,255,0)
RED = (0,0,255)
class_colors = [RED, GREEN]
# 0 without mask, 1 with mask

model = load_model( 'out/cnn.hdf5' )
preprocessors = [utils.resize(50,50), img_to_array, lambda img: img/255.0]


cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    imagenGris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(imagenGris, 1.3, 8)
    face = frame
    for x,y,w,h in faces:
        # Preprocesar imagen antes de predecir
        face = frame[y:y+h, x:x+w]

        face = utils.preprocess_single(face, preprocessors)

        # Obtener la clase con mayor probabilidad
        predictions = model.predict(face)
        label=np.argmax( predictions, axis=1)[0]

        # Dibujar rectangulo
        cv2.rectangle(frame, (x,y), (x+w, y+h), class_colors[label], 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
