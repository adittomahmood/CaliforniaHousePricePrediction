# Position Salary Prediction with Random Forest Regression

## Project Overview

This project demonstrates the application of **Random Forest Regression** to predict salaries based on position levels. The model effectively handles the non-linear relationship between position levels and salaries, particularly capturing the sharp salary jump at higher position levels.

## Features

- **Data Preprocessing**: Extract and prepare features from CSV data
- **Model Training**: Train a Random Forest Regression model with optimal parameters
- **Visualization**: Create high-resolution plots to visualize the regression approach
- **Prediction**: Make accurate salary predictions for new position levels

## Dataset

The dataset contains information about **position levels and corresponding salaries** with:

- **Position Level** (independent variable)
- **Salary** (dependent variable)
- Notable non-linear pattern with a sharp salary jump at level 10

## 🔧 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/adittomahmood/Salary_Prediction_RFR.git

# Navigate to project directory
cd Salary_Prediction_RFR

# Install required packages
pip install numpy pandas matplotlib scikit-learn
```

## 💻 Code Implementation

### Random Forest Regression Implementation

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Position level
y = dataset.iloc[:, -1].values    # Salary

# Train Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Make a prediction for position level 6.5
regressor.predict([[6.5]])

# Visualize Random Forest Regression results with higher resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

## Results

The Random Forest Regression model demonstrates excellent performance in predicting salaries based on position levels:

- **Ensemble Learning Advantage**: Leverages multiple decision trees (10 in this implementation) to model complex non-linear relationships with high accuracy
- **Step-like Predictions**: Creates a distinctive step-like prediction curve that effectively captures salary jumps
- **Robust Performance**: Handles the sharp salary increases at higher position levels without overfitting
- **No Feature Scaling Required**: Unlike some other regression techniques, Random Forest works well without feature scaling
- **Built-in Feature Importance**: Provides insights into which features most significantly impact salary predictions

## Visualization

### Random Forest Regression Visualization

_Figure 1: Random Forest Regression model with 10 trees showing step-like prediction curve_

![Random Forest Results](https://i.ibb.co.com/27Qxp9RY/Screenshot-2025-05-07-225629.png)

## What I Learned

Through this project, I've gained practical experience with:

- **Random Forest Regression** as a powerful ensemble learning method
- **Decision tree ensembles** for handling complex non-linear relationships
- **Hyperparameter tuning** for Random Forest models (n_estimators, random_state)
- **Higher resolution visualization** techniques for better model interpretation
- The significance of **sample density** in grid visualization (using 0.01 step size)
- How ensemble methods can **reduce overfitting** while maintaining predictive power
- **Step function approximation** of continuous curves through tree-based methods
- **Built-in feature importance** as an advantage of tree-based models
- The balance between **bias and variance** in ensemble models
- Visualizing ensemble model performance

## Next Steps

In future projects, I plan to:

- Implement **grid search CV** to find optimal hyperparameters for Random Forest
- Fine-tune Random Forest parameters like **n_estimators**, **max_depth**, and **min_samples_split**
- Compare with other ensemble methods like **Gradient Boosting** and **Extra Trees**
- Add **cross-validation** for more robust model evaluation
- Explore **feature importance** analysis from Random Forest models
- Implement **bootstrap aggregating (bagging)** techniques to further improve model performance
- Apply **out-of-bag error** estimation for model validation
- Test the model with **additional features** beyond position level
- Explore the effect of **random state** on model stability
- Apply these techniques to larger, more complex datasets

## Contact

Feel free to connect with me:

- GitHub: [@Aditto Mahmood](https://github.com/adittomahmood)
- LinkedIn: [Aditto Mahmood](https://linkedin.com/in/adittomahmood)
