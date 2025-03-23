import cv2
from deepface import DeepFace
import pandas as pd

# Iniciamos la cámara
cap = cv2.VideoCapture(0)

print("Presioná ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analizamos el frame para detectar emociones
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        
        face = result[0]['region']
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Mostramos la emoción en pantalla
        cv2.putText(frame, f'Emocion: {emotion}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        cv2.putText(frame, 'No se pudo detectar rostro', (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Detección de emociones - DeepFace', frame)

    # Presionar ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break



# Liberamos recursos
cap.release()
cv2.destroyAllWindows()