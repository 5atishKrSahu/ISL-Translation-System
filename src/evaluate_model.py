import pickle
from sklearn.metrics import classification_report

with open("../landmark_dataset.pkl","rb") as f:
    dataset = pickle.load(f)

with open("../models/isl_model.pkl","rb") as f:
    model = pickle.load(f)

X = dataset["data"]
y = dataset["labels"]

pred = model.predict(X)

print(classification_report(y,pred))