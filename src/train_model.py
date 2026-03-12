import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open("../landmark_dataset.pkl","rb") as f:
    dataset = pickle.load(f)

X = dataset["data"]
y = dataset["labels"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,shuffle=True,stratify=y
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("Accuracy:",acc)

with open("../models/isl_model.pkl","wb") as f:
    pickle.dump(model,f)