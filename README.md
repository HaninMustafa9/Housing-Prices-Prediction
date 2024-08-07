# Housing Prices Prediction

## Project Overview

This project aims to predict housing prices using features from a dataset. It involves data preprocessing, encoding categorical variables, scaling features, and training a linear regression model to predict the target variable, `median_house_value`.

## Data Preprocessing

### Import Libraries

The following libraries are used:
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **matplotlib**: For visualization
- **seaborn**: For statistical visualization
- **sklearn**: For machine learning tasks

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
```

### Load the Data

The dataset is loaded from a CSV file:

```python
Housing_Data = pd.read_csv("housing.csv")
```

### Exploratory Data Analysis (EDA)

1. **Describe the Data**

   ```python
   Housing_Data.head()
   Housing_Data.tail()
   Housing_Data.shape
   Housing_Data.info()
   Housing_Data.describe()
   Housing_Data.duplicated()
   ```

2. **Visualizations**

   Visualize the distribution and boxplot of features:

   ```python
   fig, axes = plt.subplots(9, 2, figsize=(15, 25))
   # Plot code here
   plt.tight_layout()
   plt.show()
   ```

### Data Cleaning

**Removing Outliers**

Outliers are removed based on the interquartile range (IQR) for multiple features:

```python
Q1_total_rooms = Housing_Data['total_rooms'].quantile(0.25)
Q3_total_rooms = Housing_Data['total_rooms'].quantile(0.75)
IQR_total_rooms = Q3_total_rooms - Q1_total_rooms
# Repeat for other features

# Remove outliers
Housing_Data_no_outliers = Housing_Data[(Housing_Data['total_rooms'] >= lower_bound_total_rooms) & (Housing_Data['total_rooms'] <= upper_bound_total_rooms) &
                    (Housing_Data['total_bedrooms'] >= lower_bound_total_bedrooms) & (Housing_Data['total_bedrooms'] <= upper_bound_total_bedrooms) &
                    (Housing_Data['population'] >= lower_bound_population) & (Housing_Data['population'] <= upper_bound_population) &
                    (Housing_Data['households'] >= lower_bound_households) & (Housing_Data['households'] <= upper_bound_households) &
                    (Housing_Data['median_income'] >= lower_bound_median_income) & (Housing_Data['median_income'] <= upper_bound_median_income)]
```

**Visualize Cleaned Data**

```python
fig, axes = plt.subplots(9, 2, figsize=(15, 25))
# Plot code here
plt.tight_layout()
plt.show()
```

### Data Transformation

**Encoding Categorical Variables**

```python
Encoded_Housing_Data = pd.get_dummies(Housing_Data_no_outliers, columns=['ocean_proximity'])
```

**Scaling Features**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Encoded_Housing_Data.drop(columns=['median_house_value']))
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Housing_Data_no_outliers['median_house_value'], test_size=0.2, random_state=42)
```

## Model Training and Evaluation

**Training the Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

**Performance Metrics**

```python
print("Training Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
print("Testing Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
print("Training R^2 Score:", model.score(X_train, y_train))
print("Testing R^2 Score:", model.score(X_test, y_test))
```

## Conclusion

The project demonstrates the process of preparing data, training a linear regression model, and evaluating its performance to predict housing prices based on various features.
