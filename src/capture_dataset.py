import cv2
import os

DATASET_PATH = "../dataset"
SAMPLES_PER_CLASS = 300

label = input("Enter new sign label: ").upper()

path = os.path.join(DATASET_PATH, label)
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = len(os.listdir(path))

print(f"Collecting images for sign: {label}")

while count < SAMPLES_PER_CLASS:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    cv2.putText(frame,f"{label} : {count}",(20,40),
    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Dataset Collection",frame)

    key = cv2.waitKey(1)

    if key == ord('s'):

        img_name = os.path.join(path,f"{count}.jpg")
        cv2.imwrite(img_name,frame)

        count += 1
        print("Saved",img_name)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()