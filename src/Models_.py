import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error,confusion_matrix,ConfusionMatrixDisplay,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
df=pd.read_csv("/storage/emulated/0/Download/RealEstate-Project/Real_Estate_Cleaned.csv")
fix,ax=plt.subplots(3,3,figsize=(9,9))
le=LabelEncoder()
df["State"]=le.fit_transform(df["State"])
df["PropertyType"]=le.fit_transform(df["PropertyType"])
df["SwimmingPool"]=le.fit_transform(df["SwimmingPool"])
df["Gym"]=le.fit_transform(df["Gym"])
df["Security"]=le.fit_transform(df["Security"])
df["Furnished"]=le.fit_transform(df["Furnished"])
df["Estate"]=le.fit_transform(df["Estate"])
df["Category"]=le.fit_transform(df["Category"])
#x for category classification
X1=df[["State","PropertyType","Bedrooms","Bathrooms","Toilets","SizeSqm","Floors","AgeYears","ParkingSpots","SwimmingPool","Gym","Security","Furnished","Estate","YearBuilt","Total_str","Price_index","Price_Size","Age_distance","Floor_parking_index","Price_INdex","Comfort"]]
#x for linear regression 
X2=df[["State","PropertyType","Bedrooms","Bathrooms","Toilets","SizeSqm","Floors","AgeYears","ParkingSpots","SwimmingPool","Gym","Security","Furnished","Estate","YearBuilt","Total_str","Age_distance","Floor_parking_index","Comfort"]]
#y for category classification
y_category=df["Category"]
#y for linear regression
y_price=df["Price(NGN)"]

X2_train,X2_test,y_price_train,y_price_test=train_test_split(X2,y_price,test_size=0.3,random_state=7)

X1_train,X1_test,y_category_train,y_category_test=train_test_split(X1,y_category,test_size=0.3,random_state=7)

R_model=LinearRegression()
R_model.fit(X2_train,y_price_train)
y_price_pred=R_model.predict(X2_test)
print(f"R²: {r2_score(y_price_test,y_price_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_price_test,y_price_pred))}")
#imballance correction using smote
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=7)
X1_train_res, y_category_train_res = sm.fit_resample(X1_train, y_category_train)
#Logistic Regression
model=LogisticRegression(max_iter=1000)
model.fit(X1_train_res,y_category_train_res)
y_pred=model.predict(X1_test)
print(accuracy_score(y_pred,y_category_test))
print(classification_report(y_pred,y_category_test))
cm=confusion_matrix(y_pred,y_category_test)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=("Luxury","Budget","Mid-Range"))
disp.plot(ax=ax[0,0],cmap='Blues')
#Decision tree classifier
model1=DecisionTreeClassifier(max_depth=7,random_state=7)
model1.fit(X1_train,y_category_train)
y_pred1=model1.predict(X1_test)
print(accuracy_score(y_pred1,y_category_test))
print(classification_report(y_pred1,y_category_test))
cm1=confusion_matrix(y_pred1,y_category_test)
disp=ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=("Luxury","Budget","Mid-Range"))
disp.plot(ax=ax[0,1],cmap='Blues')
#Random forest classifier
model2=RandomForestClassifier(n_estimators=120,random_state=42,class_weight="balanced")
model2.fit(X1_train_res,y_category_train_res)
y_pred2=model2.predict(X1_test)
print(accuracy_score(y_pred2,y_category_test))
print(classification_report(y_pred2,y_category_test))
cm2=confusion_matrix(y_pred2,y_category_test)
disp=ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=("Luxury","Budget","Mid-Range"))
#feature importance
importance=model2.feature_importances_
features=X1.columns
im_df=pd.DataFrame({"Feature":features, "Importance":importance})
im_df=im_df.sort_values(by="Importance")
ax[1,0].barh(im_df["Feature"],im_df["Importance"])
ax[1,0].set_xlabel("importance")
disp.plot(ax=ax[0,2],cmap='Blues')
plt.tight_layout()
plt.show()
#finetunning
params={
    "max_depth":[4,9,12,None],
    "min_samples_split":[3,5,9,12],
    "n_estimators":[50,100,150]
}
grid=GridSearchCV(RandomForestClassifier(random_state=7),params,cv=7,scoring="accuracy",verbose=1)
grid.fit(X1_train,y_category_train)
print(f"accuracyscore {grid.best_score_:.4f}")
print(f"bestparameter: {grid.best_params_}")
y_pred3=grid.best_estimator_.predict(X1_test)
accuracy3=accuracy_score(y_category_test,y_pred3)
classification3=classification_report(y_category_test,y_pred3)
print(f"FinetunnedRandomForest(accuracy): {accuracy3:.4f}")
print(f"FinetunnedRandomForest(classifcationreport): {classification3}")
#saving the best model
pickle.dump(model2,open("/storage/emulated/0/Download/RealEstate-Project/classifier_model.pkl","wb"))
pickle.dump(R_model,open("/storage/emulated/0/Download/RealEstate-Project/Linear_model.pkl","wb")) 
