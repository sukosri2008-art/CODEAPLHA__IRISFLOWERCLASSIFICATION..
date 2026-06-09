import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\srija\Downloads\iris.csv")
print(data.head())
X=data.drop("species",axis=1)
y=data["species"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print("REAL VALUE:",y_test)
accuracy=accuracy_score(y_test,prediction)
print("ACCURACY SCORE:",accuracy*100)
for i in range (5):
    print("Actual : ",y_test.iloc[i],
          " ||  Predicted : ",prediction[i])
