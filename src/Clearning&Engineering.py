import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/storage/emulated/0/Download/nigerian_realestate_messy.csv")
df=df.drop_duplicates()
df["State"]=df["State"].astype(str).str.strip()
df["Area"]=df["Area"].astype(str).str.strip()
df["PropertyType"]=df["PropertyType"].astype(str).str.strip()
df["Bedrooms"]=np.abs(df["Bedrooms"].clip(1,6))
df["Bedrooms"]=df["Bedrooms"].fillna(df.groupby("PropertyType")["Bedrooms"].transform(lambda x:x.mode()[0]))
df["Bedrooms"]=df["Bedrooms"].astype(int)
df["Bathrooms"]=np.abs(df["Bedrooms"].clip(1,7))
df["Bathrooms"]=df["Bathrooms"].astype(int)
df["Toilets"]=np.abs(df["Toilets"].clip(1,7))
df["Toilets"]=df["Toilets"].astype(int)
df["SizeSqm"]=df["SizeSqm"].astype(str).str.replace("m2","").str.replace("sqm","").str.replace("-","").str.replace("sq.ft","").str.strip()
df["SizeSqm"]=pd.to_numeric(df["SizeSqm"],errors="coerce")
max1=(df["SizeSqm"].quantile(0.75)+1.5*(df["SizeSqm"].quantile(0.75)-df["SizeSqm"].quantile(0.25))).round(1)
min1=(df["SizeSqm"].quantile(0.25)-1.5*(df["SizeSqm"].quantile(0.75)-df["SizeSqm"].quantile(0.25))).round(1)
df["SizeSqm"]=df["SizeSqm"].clip(min1,max1)
df["SizeSqm"]=(df["SizeSqm"].fillna(df.groupby(["PropertyType","Bedrooms"])["SizeSqm"].transform("mean"))).round(1)
df["Floors"]=(np.abs(df["Toilets"])).astype(int)
df["AgeYears"]=df["AgeYears"].astype(str).str.replace("new","0").str.replace("yrs","").str.replace("years","").str.strip()
df["AgeYears"]=pd.to_numeric(df["AgeYears"],errors="coerce")
df["AgeYears"]=df["AgeYears"].apply(lambda x: int(x/10) if x>50 else x)
df["AgeYears"]=df["AgeYears"].fillna(df.groupby(["Area","PropertyType"])["AgeYears"].transform("mean"))
df["AgeYears"]=df["AgeYears"].astype(int)
df["ParkingSpots"]=(np.abs(df["ParkingSpots"])).astype(int)
df["SwimmingPool"]=df["SwimmingPool"].astype(str).str.strip()
df["Gym"]=df["Gym"].astype(str).str.strip()
df["Security"]=df["Security"].astype(str)
security_correction={"24/7":"Always","24hrs":"Always","Nill":"No Security","Day Only":"Day Only"}
df["Security"]=df["Security"].map(security_correction)
df["Security"]=df["Security"].fillna(df.groupby(["State","PropertyType"])["Security"].transform(lambda x: x.mode()[0]))
df["Furnished"]=df["Furnished"].astype(str).str.capitalize().str.strip()
furnish_correction={"F":"Furnished","Uf":"Unfurnished","Sf":"Semi-furnished"}
df["Furnished"]=df["Furnished"].replace(furnish_correction)
df["Estate"]=df["Estate"].astype(str).str.capitalize().str.strip()
estate_correction={"N":"No","Y":"Yes","1":"Yes","0":"No"}
df["Estate"]=df["Estate"].replace(estate_correction)
df["DistanceToCBD(km)"]=df["DistanceToCBD(km)"].astype(str).str.replace("km","").str.replace("-","").str.strip()
def distance_corrector(c):
    if "miles" in c:
        c=c.replace("miles","").strip()      
        k=1.609*(float(c))
        return str(k)
    else:
        return c
df["DistanceToCBD(km)"]=df["DistanceToCBD(km)"].apply(lambda x:distance_corrector(x))
df["DistanceToCBD(km)"]=pd.to_numeric(df["DistanceToCBD(km)"],errors="coerce")
df["DistanceToCBD(km)"]=df["DistanceToCBD(km)"].fillna(df.groupby(["State","Area"])["DistanceToCBD(km)"].transform("mean"))
df["DistanceToCBD(km)"]=df["DistanceToCBD(km)"].round(1)
df["YearBuilt"]=df["YearBuilt"].astype(str).str.replace("AD","").str.strip()
def yearbuilt_corrector(c):
    if "circa" in c:
        k=c.index("a")
        return c[(k+2):]
    else:
        return c
df["YearBuilt"]=df["YearBuilt"].apply(lambda x: yearbuilt_corrector(x))
df["YearBuilt"]=pd.to_numeric(df["YearBuilt"],errors="coerce")
df["YearBuilt"]=np.abs(df["YearBuilt"].apply(lambda x: x/2 if x>2025 else x))
df["YearBuilt"]=df["YearBuilt"].fillna(df.groupby(["Area","PropertyType"])["YearBuilt"].transform(lambda x: x.mode()[0]))
df["YearBuilt"]=df["YearBuilt"].astype(int)
df["Price(NGN)"]=df["Price(NGN)"].astype(str).str.replace("\"","").str.replace("NGN","").str.replace("\u20a6","").str.replace(",","").str.strip()
df["Price(NGN)"]=pd.to_numeric(df["Price(NGN)"],errors="coerce")
max2=df["Price(NGN)"].quantile(0.75)+1.5*(df["Price(NGN)"].quantile(0.75)-df["Price(NGN)"].quantile(0.25))
min2=df["Price(NGN)"].quantile(0.25)-1.5*(df["Price(NGN)"].quantile(0.75)-df["Price(NGN)"].quantile(0.25))
df["Price(NGN)"]=df["Price(NGN)"].clip(min2,max2)
df["Price(NGN)"]=df["Price(NGN)"].fillna(df.groupby(["State","PropertyType","Furnished"])["Price(NGN)"].transform("mean"))
df["Price(NGN)"]=df["Price(NGN)"].round(1)
df["Category"]=df["Category"].astype(str).str.capitalize().str.strip()
category_corection={"High-end":"Luxury","Nan":np.nan,"Premium":"Luxury","L":"Luxury","Low-end":"Budget","Budget":"Budget","B":"Budget","M":"Mid-Range","Mid":"Mid-Range","Affordable":"Budget","Mid-range":"Mid-Range","Middle":"Mid-Range"}
df["Category"]=df["Category"].replace(category_corection)
df["Total_str"]=(df["Toilets"]+df["Bathrooms"]+df["Bedrooms"])
df["Price_index"]=(df["Price(NGN)"]/df["SizeSqm"]).round(3)
df["Price_Size"]=(df["Price(NGN)"]/df["SizeSqm"]).round(3)
df["Age_distance"]=(df["AgeYears"]+df["DistanceToCBD(km)"]).round(2)
df["Floor_parking_index"]=(0.5*(df["Floors"]+df["ParkingSpots"])+df["SizeSqm"]).round(3)
df["Price_INdex"]=(df["Price(NGN)"]+df["SizeSqm"])/((df["Bathrooms"]+df["Bedrooms"]+df["Floors"]+df["Toilets"])).round(3)
df["Comfort"]=(df["Bedrooms"]/(df["Toilets"]+df["Bathrooms"])).round(3)
df.to_csv("Real_Estate_Cleaned.csv",index=False)
