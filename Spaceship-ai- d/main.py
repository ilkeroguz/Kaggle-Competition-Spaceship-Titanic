import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

train_set = pd.read_csv("data/train.csv")

#boş değer içeren satırları doldur ve kontrol et
train_set[["Cabin", "HomePlanet", "CryoSleep", "Destination", "VIP"]] = train_set[["Cabin", "HomePlanet", "CryoSleep", "Destination", "VIP"]].fillna(method= "pad")
train_set[["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]] = train_set[["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]].fillna(value= train_set[["Age","RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]].mean())
train_set["Name"] = train_set["Name"].fillna(value= "None")
print(train_set.isnull().sum())

#bool değerleri int'e çevir
train_set["CryoSleep"] = train_set["CryoSleep"].astype(int)
train_set["VIP"] = train_set["VIP"].astype(int)
train_set["Transported"] = train_set["Transported"].astype(int)

#kategorizasyon içeren sütunları label encoder ile encodele
encoder = LabelEncoder()
for column in train_set:
    train_set[column] = encoder.fit_transform(train_set[column])
    
#train_set["Destination"] = encoder.fit_transform(train_set["Destination"])
#train_set["HomePlanet"] = encoder.fit_transform(train_set["HomePlanet"])
#train_set["Cabin"] = encoder.fit_transform(train_set["Cabin"])

# X ve y 
X = train_set.drop(columns=['Transported', "Name"])
y = train_set["Transported"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, 
    random_state=43,
    stratify= y,
    test_size= 0.5,
)

model = LogisticRegression(max_iter=300,penalty="l1" ,class_weight="balanced",solver="liblinear", C = 0.9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("porno")
joblib.dump(model, "model.pkl")