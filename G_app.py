
import os
import numpy as np
import pickle
from flask import Flask,request,render_template
#loading linear model
R_model=pickle.load(open("Linear_model.pkl","rb"))
#loading classifier best model
C_model=pickle.load(open("classifier_model.pkl","rb"))
app=Flask( __name__ )
@app.route("/",methods=["GET","POST"])
def work():
    price=None
    category=None
    if request.method=="POST":
        State=float(request.form["state"])
        PropertyType=float(request.form["type"])
        Bedrooms=float(request.form["bedrooms"])
        Bathrooms=float(request.form["bathrooms"])
        Toilets=float(request.form["toilets"])
        SizeSqm=float(request.form["size"])
        Floors=float(request.form["floors"])
        AgeYears=float(request.form["age"])
        ParkingSpots=float(request.form["parkingspot"])
        SwimmingPool=float(request.form["pool"])
        Gym=float(request.form["gym"])
        Security=float(request.form["security"])
        Furnished=float(request.form["furnished"])
        Estate=float(request.form["estate"])
        YearBuilt=float(request.form["year"])
        DistancetoCBD=float(request.form["distance"])
        Total_str=Toilets+Bathrooms+Bedrooms
        Age_distance=AgeYears+DistancetoCBD
        Floor_parking_index=(0.5*(Floors+ParkingSpots)+SizeSqm)
        Comfort=Bedrooms/(Toilets+Bathrooms)
        feature1=np.array([[State,PropertyType,Bedrooms,Bathrooms,Toilets,SizeSqm,Floors,AgeYears,ParkingSpots,SwimmingPool,Gym,Security,Furnished,Estate,YearBuilt,Total_str,Age_distance,Floor_parking_index,Comfort]])
        price=(R_model.predict(feature1)[0]).round(1)
        Price_index=price/SizeSqm
        Price_Size=price/SizeSqm
        Price_INdex=(price+SizeSqm)/(Bathrooms+Bedrooms+Floors+Toilets)
        feature2=np.array([[State,PropertyType,Bedrooms,Bathrooms,Toilets,SizeSqm,Floors,AgeYears,ParkingSpots,SwimmingPool,Gym,Security,Furnished,Estate,YearBuilt,Total_str,Price_index,Price_Size,Age_distance,Floor_parking_index,Price_INdex,Comfort]])
        result=C_model.predict(feature2)[0]
        if result==2:
            category="Mid-Range"
        elif result==1:
            category="Luxury"
        else:
            category="Low-Budget"
    return render_template("design.html",price=price,category=category)
if __name__==("__main__"):
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) 
      
