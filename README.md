# 🏠 Nigerian Real Estate Price & Category Predictor

A complete dual-target ML project predicting both the **estimated property price in NGN** (regression) and **property category** (classification) from Nigerian real estate data — using a chained model architecture.

## 🌐 Live Demo
**[Try the app →](https://web-real-estate-pricce-category-predictor.up.railway.app)**

---

## 📌 Project Overview
The Nigerian real estate market is one of the most opaque in Africa — property prices range from ₦3 million to ₦500 million with little transparency. This system predicts both the price AND category of a property simultaneously, giving buyers, sellers and agents an objective data-driven estimate.

**What makes this architecturally unique:**

This project uses a **chained model architecture** — Model 1 predicts the price, and that predicted price becomes an input feature for Model 2. This creates a meaningful relationship: a property priced at ₦45M should logically be classified as Mid-Range, and the model uses this predicted price to make a more informed category decision.

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 2,035 (raw) → 2,000 (after cleaning) |
| Columns | 20 (raw) → 24 (after engineering) |
| Regression Target | Price(NGN) |
| Classification Target | Category: Budget / Mid-Range / Luxury |

---

## 🧹 Data Cleaning Challenges
| Column | Problem | Solution |
|---|---|---|
| Price(NGN) | "NGN52,548,000", "₦62,626,000", outliers ×1000 | Strip NGN/₦, commas, IQR clip |
| SizeSqm | "180 sqm", "180sq.ft", "180m2" — 3 unit formats | Strip all unit suffixes |
| AgeYears | "8 years", "5yrs", "new" (means 0), outliers ×10 | "new"→0, strip suffixes, IQR clip |
| DistanceToCBD | "5.2km", "5.2 km", "3.2miles" — mixed units! | Miles × 1.60934 → km |
| YearBuilt | "2015AD", "circa 2019" — string formats | Strip "AD", extract numeric |
| Category | 18 different formats across 3 classes | Comprehensive dictionary map |
| Duplicates | 35 hidden rows | drop_duplicates() |

---

## ⚙️ Feature Engineering
| Feature | Formula | Meaning |
|---|---|---|
| Total_str | Toilets + Bathrooms + Bedrooms | Overall facility count — property quality proxy |
| Age_distance | AgeYears + DistanceToCBD | Combined depreciation: older AND farther = lower value |
| Floor_parking_index | 0.5×(Floors + ParkingSpots) + SizeSqm | Weighted space and amenity score |
| Comfort | Bedrooms / (Toilets + Bathrooms) | Lower ratio = more bathrooms per bedroom = premium |

---

## 🔑 The Log Transformation Discovery
```
Without log transform: R² = 0.24
With np.log1p(Price):  R² = 0.72
```
Nigerian property prices follow a heavily skewed exponential distribution. Applying log transformation normalized the target variable dramatically — improving R² by 3×. In Flask, predictions are reversed using `np.expm1()`.

---

## 🤖 Chained Model Architecture

```
Raw Features
     ↓
Model 1 (RandomForestRegressor)
     ↓
Predicted Price (NGN)
     ↓
Raw Features + Predicted Price
     ↓
Model 2 (RandomForestClassifier)
     ↓
Budget / Mid-Range / Luxury
```

### Model Results
| Model | Type | Score | Notes |
|---|---|---|---|
| Linear Regression | Regression | R²=0.18 | Baseline |
| RandomForestRegressor | Regression | R²=0.24 | Better but skewed |
| **RF + Log Transform** | **Regression** | **R²=0.72** | **Log = key improvement** |
| Logistic Regression | Classification | 71% | Weak boundary |
| Decision Tree | Classification | 82% | Improved with price feature |
| Random Forest | Classification | 91% | Strong ensemble |
| **RF Tuned + Chained** | **Classification** | **95%** | **Champion ✅** |

---

## 🌐 Flask Dual Prediction
```python
# Step 1 — Price prediction
predicted_log_price = price_model.predict(X_features)[0]
predicted_price = np.expm1(predicted_log_price)

# Step 2 — Compute price-derived features
Price_index = predicted_price / Total_str
Price_Size = predicted_price / SizeSqm

# Step 3 — Category prediction using predicted price
predicted_category = cat_model.predict(X_with_price)[0]
```

---

## 🏗️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn (RandomForest, GridSearchCV)
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3 (Dark Glass-morphism Gold Theme)
- **Deployment:** Railway.app
- **Version Control:** GitHub

---

## 📁 Project Structure
```
RealEstatePredictor/
├── data/
│   └── nigerian_realestate_messy.csv
├── models/
│   ├── price_model.pkl
│   └── category_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/RealEstate_Price-Category_Predictor
cd RealEstate_Price-Category_Predictor
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Learnings
1. **Log transformation** — np.log1p() improved R² from 0.24 to 0.72 on skewed price data
2. **Chained models** — predicted price from Model 1 feeds as feature into Model 2
3. **np.expm1()** — exact inverse of np.log1p(), restores actual NGN price in Flask
4. **Miles → km** — DistanceToCBD required 1.60934 conversion factor
5. **"new" → 0** — AgeYears column had "new" meaning zero years old
6. **SizeSqm** — ranked #1 in feature importance, confirming size as primary value driver

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — built during Industrial Training placement.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)

