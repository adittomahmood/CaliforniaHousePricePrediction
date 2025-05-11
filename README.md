# California House Price Prediction

## Project Overview

This project demonstrates the application of machine learning techniques to predict house prices in California based on various features. Two regression models are implemented and compared: **Linear Regression** and **Random Forest Regression**.

## Features

- **Data Acquisition**: Direct download from Kaggle using API
- **Exploratory Data Analysis**: Comprehensive visualization of data distributions and relationships
- **Feature Engineering**: Creation of meaningful derived features to improve model performance
- **Data Preprocessing**: Pipeline implementation with imputation and standardization
- **Model Training**: Implementation of Linear Regression and Random Forest models
- **Performance Evaluation**: Comprehensive metrics comparison (RMSE, MAE, R²)
- **Visualization**: Insightful plots showing model performance and feature importance

## Dataset

The dataset contains information about California housing districts with features including:
- Geographic information (longitude, latitude)
- Housing characteristics (total_rooms, total_bedrooms, households)
- Population demographics (population, median_income)
- Ocean proximity
- Median house value (target variable)

## Installation & Setup

```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn kaggle

# Set up Kaggle API credentials
# (Requires uploading kaggle.json to /root/.kaggle/)

# Download the dataset
kaggle datasets download -d shibumohapatra/house-price
```

## Code Implementation

### Data Acquisition and Preparation

```python
# Set up Kaggle credentials
import os
import shutil
os.makedirs('/root/.kaggle', exist_ok=True)
shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 600)

# Download dataset
!pip install -q kaggle
!kaggle datasets download -d shibumohapatra/house-price
!unzip -q house-price.zip

# Load dataset
df = pd.read_csv("/content/1553768847-housing.csv")
```

### Data Preprocessing

```python
# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].mean(), inplace=True)

# Encode categorical features
categories = list(df["ocean_proximity"].unique())
ocean_map = {category: idx for idx, category in enumerate(categories)}
df["ocean_proximity_encoded"] = df["ocean_proximity"].map(ocean_map)
df = df.drop(columns=["ocean_proximity"])
```

### Feature Engineering

```python
# Create derived features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
df["households_per_population"] = df["households"] / df["population"]
df["rooms_per_person"] = df["total_rooms"] / df["population"]
df["income_per_household"] = df["median_income"] / df["households"]
df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
df["avg_household_size"] = df["population"] / df["households"]
```

### Data Visualization

```python
# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Geographical visualization
df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
        s=df["population"] / 100, label="population",
        c="median_house_value", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))
plt.show()

# Income category distribution
df["income_cat"] = pd.cut(df["median_income"],
                          bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                          labels=[1, 2, 3, 4, 5])
df["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
```

### Model Training and Evaluation

```python
# Split data with stratification for unbiased sampling
strat_train_set, strat_test_set = train_test_split(
    df, test_size=0.2, stratify=df["income_cat"], shuffle=True, random_state=1215)

# Create preprocessing pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler())
])

# Train Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(train_prepared, train_labels)

# Train Random Forest model
forest_reg = RandomForestRegressor(n_estimators=100, random_state=1215)
forest_reg.fit(train_features, train_labels)

# Evaluate models
# Linear Regression metrics
test_mse = mean_squared_error(test_labels, test_predictions)
test_rmse = test_mse ** 0.5
test_mae = mean_absolute_error(test_labels, test_predictions)
test_r2 = r2_score(test_labels, test_predictions)

# Random Forest metrics
forest_mse = mean_squared_error(test_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_mae = mean_absolute_error(test_labels, forest_predictions)
forest_r2 = r2_score(test_labels, forest_predictions)
```

## Results

The project reveals the significant advantage of Random Forest Regression over Linear Regression for this dataset:

### Model Performance Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|---|
| Linear Regression | 73680.88 | 54188.55 | 0.6050 |
| Random Forest | 60439.95 | 41949.50 | 0.7342 |

### Key Findings

- **Random Forest**: The Random Forest model outperforms Linear Regression across all metrics, with significantly lower error rates (RMSE, MAE) and higher R² value
- **Feature Importance**: The Random Forest model identified median_income, ocean_proximity_encoded, and population_per_household as the most influential features
- **Non-linearity**: The superior performance of Random Forest indicates strong non-linear relationships in the data that linear models cannot capture

## Visualizations

### Correlation Heatmap
![Correlation Heatmap](https://i.ibb.co.com/NgYpngjK/HeatMap.png)

### Geographic Distribution of House Prices
![Geographic Distribution](https://i.ibb.co.com/xKQPs6Hj/Screenshot-2025-05-11-142611.png)

### Income Category Distribution
![Income Category Distribution](https://i.ibb.co.com/FbfKRL1b/income-category.png)

### Actual vs Predicted Prices (Linear Regression)
![Actual vs Predicted](https://i.ibb.co.com/G4L6jMrt/Actual-vs-Predicted-Price-LR.png)

### Actual vs Predicted Prices (Random Forest)
![Actual vs Predicted](https://i.ibb.co.com/LdkYDnf9/Actual-vs-Predicted-Price-RF.png)

### Model Performance Comparison
![Model Comparison](https://i.ibb.co.com/M5SWLPFW/Feature-Importance.png)

### Feature Importance
![Feature Importance]()

## What I Learned

Through this project, I've gained practical experience with:

- **Kaggle API Integration** 
- **Feature Engineering** 
- **Stratified Sampling** 
- **Pipeline Implementation** 
- **Ensemble Methods** 
- **Model Evaluation**
- **Data Visualization** 
- **Geospatial Analysis** 

## Next Steps

In future iterations of this project, I plan to:

- Implement **Hyperparameter Tuning** using Grid Search or Random Search
- Explore **Advanced Ensemble Methods** like Gradient Boosting and XGBoost
- Apply **Cross-Validation** for more robust model evaluation
- Perform **Feature Selection** to identify the most predictive variables
- Implement **Deep Learning Models** to capture even more complex patterns
- Create an **Interactive Dashboard** for exploring predictions
- Deploy the model as a **Web Application** for real-time predictions
- Incorporate **Additional External Data** sources to enrich the feature set
- Explore **Clustering Techniques** to identify similar housing districts

## Contact

Feel free to connect with me:

- GitHub: [Your GitHub Username]
- LinkedIn: [Your LinkedIn Profile]
