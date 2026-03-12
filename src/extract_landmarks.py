import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

DATASET_DIR = "../dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

data = []
labels = []

for label in os.listdir(DATASET_DIR):

    folder = os.path.join(DATASET_DIR,label)

    for img_path in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,img_path))
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                landmark_list = []

                for lm in hand.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                data.append(landmark_list)
                labels.append(label)

print("Saving landmark dataset...")

dataset = {
    "data":np.array(data),
    "labels":np.array(labels)
}

with open("../landmark_dataset.pkl","wb") as f:
    pickle.dump(dataset,f)