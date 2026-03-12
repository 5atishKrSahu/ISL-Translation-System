import cv2
import mediapipe as mp
import numpy as np
import pickle

with open("../models/isl_model.pkl","rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            landmark_list = []

            for lm in hand.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            prediction = model.predict([landmark_list])

            cv2.putText(frame,prediction[0],(50,50),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("ISL Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()