# Importing the liabraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import Imputer
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

#setting working directory
os.chdir("D:/Data Science Edwisor/6. Projects/2. Project Two Bike Rental Count")

# Importing the dataset
dataset = pd.read_csv('day.csv')

# Exploratory Data Analysis
# Converting dteday to proper format
dataset['dteday'] = pd.to_datetime(dataset['dteday'])
# Extracting the date
dataset['date'] = dataset['dteday'].dt.day

# Missing Value Analysis
dataset.isnull().sum()
# No missing values found

# Converting to proper datatypes
dataset['season'] = dataset['season'].astype('object')
dataset['yr'] = dataset['yr'].astype('object')
dataset['mnth'] = dataset['mnth'].astype('object')
dataset['holiday'] = dataset['holiday'].astype('object')
dataset['weekday'] = dataset['weekday'].astype('object')
dataset['workingday'] = dataset['workingday'].astype('object')
dataset['weathersit'] = dataset['weathersit'].astype('object')

# Outlier Analysis for numeric valriables.
# 1. temp
plt.boxplot(dataset['temp'])

# 2. atemp
plt.boxplot(dataset['atemp'])

# 3. hum
plt.boxplot(dataset['hum'])

# 4. windspeed
plt.boxplot(dataset['windspeed'])

# 5. casual
plt.boxplot(dataset['casual'])

# 6. registered
plt.boxplot(dataset['registered'])

# 7. date
plt.boxplot(dataset['date'])

# variables hum, windspeed, casual have outliers present in them so we try to fix it.
cnames = ['hum', 'windspeed', 'casual']
for i in cnames:
    print(i)
    q75, q25 = np.percentile(dataset.loc[:,i], [75,25])
    #in the above stmnt we are calculating the [75,25]th percentiles and storing them in q75 and q25 respectively.
    iqr = q75 - q25
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)
    print(mini)
    print(maxi)
    dataset.loc[dataset.loc[:, i] < mini, i] = np.nan
    dataset.loc[dataset.loc[:, i] > maxi, i] = np.nan

dataset.isnull().sum()

"""# finding the best method for imputing the missing values
dataset['casual'].iloc[7]
# Actual value is 68
dataset['casual'].iloc[7] = np.nan
# Mean
dataset['casual'] = dataset['casual'].fillna(dataset['casual'].mean())
dataset['casual'].iloc[7]
# Mean value = 732.978

# Median
dataset['casual'] = dataset['casual'].fillna(dataset['casual'].median())
dataset['casual'].iloc[7]
# Median value = 675"""
# finalising median for imputation
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(dataset.iloc[:, [11,12,13]])
dataset.iloc[:, [11,12,13]] = imputer.transform(dataset.iloc[:, [11,12,13]])
dataset.isnull().sum()
# Recheck
# 1. hum
plt.boxplot(dataset['hum'])

# 2. windspeed
plt.boxplot(dataset['windspeed'])

# 3. casual
plt.boxplot(dataset['casual'])
# variables windspeed and casual have some outliers again so we try to remove them
cnames = ['windspeed', 'casual']
for i in cnames:
    print(i)
    q75, q25 = np.percentile(dataset.loc[:,i], [75,25])
    #in the above stmnt we are calculating the [75,25]th percentiles and storing them in q75 and q25 respectively.
    iqr = q75 - q25
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)
    print(mini)
    print(maxi)
    dataset.loc[dataset.loc[:, i] < mini, i] = np.nan
    dataset.loc[dataset.loc[:, i] > maxi, i] = np.nan

dataset.isnull().sum()
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(dataset.iloc[:, [12,13]])
dataset.iloc[:, [12,13]] = imputer.transform(dataset.iloc[:, [12,13]])

# Recheck
# 1. windspeed
plt.boxplot(dataset['windspeed'])

# 2. casual
plt.boxplot(dataset['casual'])

# variable casual has still some outliers left trying to remove them
cnames = ['casual']
for i in cnames:
    print(i)
    q75, q25 = np.percentile(dataset.loc[:,i], [75,25])
    #in the above stmnt we are calculating the [75,25]th percentiles and storing them in q75 and q25 respectively.
    iqr = q75 - q25
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)
    print(mini)
    print(maxi)
    dataset.loc[dataset.loc[:, i] < mini, i] = np.nan
    dataset.loc[dataset.loc[:, i] > maxi, i] = np.nan

dataset.isnull().sum()
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(dataset.iloc[:, [13]])
dataset.iloc[:, [13]] = imputer.transform(dataset.iloc[:, [13]])
# Recheck
plt.boxplot(dataset['casual'])

# Removing the remaining outliers
cnames = ['casual']
for i in cnames:
    print(i)
    q75, q25 = np.percentile(dataset.loc[:,i], [75,25])
    #in the above stmnt we are calculating the [75,25]th percentiles and storing them in q75 and q25 respectively.
    iqr = q75 - q25
    mini = q25 - (iqr*1.5)
    maxi = q75 + (iqr*1.5)
    print(mini)
    print(maxi)
    dataset.loc[dataset.loc[:, i] < mini, i] = np.nan
    dataset.loc[dataset.loc[:, i] > maxi, i] = np.nan

dataset.isnull().sum()
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(dataset.iloc[:, [13]])
dataset.iloc[:, [13]] = imputer.transform(dataset.iloc[:, [13]])
# Recheck
plt.boxplot(dataset['casual'])

# Feature Selection
# Correlation Analysis(between numeric variables)
subset = dataset.iloc[:, [9,10,11,12,13,14,16]]

f, ax = plt.subplots(figsize = (7, 5))
corr = subset.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
           square = True, ax = ax)
# Variable temp and atemp are highly correlated.

# Chi square test for categorical variable
# 1. season
cat_names = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['season'], dataset[i]))
    print(p)

# variable season month and weathersit are dependent

# 2. yr
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['yr'], dataset[i]))
    print(p)

# 3. mnth
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['mnth'], dataset[i]))
    print(p)

# 4. holiday
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['holiday'], dataset[i]))
    print(p)
# holiday, weekday and workingday are found to be dependent

# 5. weekday
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['weekday'], dataset[i]))
    print(p)

# 6. workingday
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['workingday'], dataset[i]))
    print(p)

# 7. weathersit
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(dataset['weathersit'], dataset[i]))
    print(p)

# from above we see that variable temp and atemp are correlated, variables
# season, month and weathersit are dependent to eachother and variables holiday,
# weekday and workingday are also dependent on eachother.
# We keep temp, weathersit and holiday and remove rest of the dependent variables from our
# dataset.

# Removing not required variables from the dataset.
dataset = dataset.iloc[:, [3,5,8,9,11,12,13,14,15,16]]

# Visualisations
# 1. yr
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['yr'], y = dataset['cnt'], s=10)
plt.xlabel('Year')
plt.ylabel('Bike_Count')
plt.show()

# 2. holiday
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['holiday'], y = dataset['cnt'], s=10)
plt.xlabel('Holiday')
plt.ylabel('Bike_Count')
plt.show()

# weathersit
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['weathersit'], y = dataset['cnt'], s=10)
plt.xlabel('Weather_Situation')
plt.ylabel('Bike_Count')
plt.show()

# temp
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['temp'], y = dataset['cnt'], s=10)
plt.xlabel('Temperature')
plt.ylabel('Bike_Count')
plt.show()

# hum
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['hum'], y = dataset['cnt'], s=10)
plt.xlabel('Humidity')
plt.ylabel('Bike_Count')
plt.show()

# windspeed
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['windspeed'], y = dataset['cnt'], s=10)
plt.xlabel('Windspeed')
plt.ylabel('Bike_Count')
plt.show()

# casual
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['casual'], y = dataset['cnt'], s=10)
plt.xlabel('Casual')
plt.ylabel('Bike_Count')
plt.show()

# registered
plt.figure(figsize=(14,7))
plt.scatter(x = dataset['registered'], y = dataset['cnt'], s=10)
plt.xlabel('Registered')
plt.ylabel('Bike_Count')
plt.show()

# Normality Check
# 1. yr
for i in ['yr']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()  
# 2. holiday
for i in ['holiday']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
# 3. weathersit
for i in ['weathersit']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
# 4. temp
for i in ['temp']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
# 5. hum
for i in ['hum']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show() 
# 6. windspeed
for i in ['windspeed']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show() 
# 7. casual
for i in ['casual']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show() 
# 8. registered
for i in ['registered']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show() 
# 9. cnt
for i in ['cnt']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show() 
# 10. date
for i in ['date']:
    print(i)
    sns.distplot(dataset[i],bins='auto',color='blue')
    plt.title("Distribution for Variable "+i)
    plt.ylabel("Density")
    plt.show()
# No significant skewness found in our predictors.

# Seperating independent and dependent variables.
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,9]].values
Y = dataset.iloc[:, 8].values

# Creating dummies for categorical variables
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy variable trap
X = X[:, [1,3,5,6,7,8,9,10,11,12]]


# Creating intercept for our OLS model
X = add_constant(X)

"""# Calculate VIF factor
X = pd.DataFrame(X)
vif = pd.DataFrame()
for i in range(X.shape[1]):
    vif["VIF Factor"] = variance_inflation_factor(X.values, i)
    vif["features"] = X.columns"""

# SPLITTING THE DATA INTO TRAIN AND TEST.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Building Models

# Linear Regression Model
# OLS
X_opt = X_train[:, [0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

# Doing backward elimination for MLR
# X6 has p-value > 0.05 hence we remove it from our list of predictors which is
# index position 6 of X_train and retrain our model
X_opt = X_train[:, [0,1,2,3,4,5,7,8,9,10]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()
# X2 has the highest p-value and > 0.05 so we remove it.
X_opt = X_train[:, [0,1,3,4,5,7,8,9,10]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()
# X8 which has p-value > 0.05 has to be removed. Removing index 10 from X_train
X_opt = X_train[:, [0,1,3,4,5,7,8,9]]
regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()

# Now all the predictors have p-value < 0.05. So this is our best list of predictors.

# Now preparing the X_test set for our model predictions.
# Removing X6, X2 and X10 from our dataset.
X_test = X_test[:, [0,1,3,4,5,7,8,9]]

# Predicting on Test Set
Y_pred = regressor_OLS.predict(X_test)

"""# Storing back the results
backup = pd.DataFrame(X_test)
backup.to_csv('Sample_Input_Python.csv', index = False)
backup_pred = pd.DataFrame({'Bike_Counts': Y_pred})
backup_pred.to_csv('Sample_Output_Python.csv', index = False)"""

# Calculating Mape
def MAPE(true, pred):
    mape = np.mean(np.abs((true - pred)/true))* 100
    return mape

MAPE(Y_test, Y_pred)

# Calculating RMSE
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# Test Data
r2_score(Y_test, Y_pred)

# visualising predictions and actual values
plt.scatter(x = range(0, Y_test.size), y = Y_test, c = 'blue', label = 'Actual', alpha = 0.3)
plt.scatter(x = range(0, Y_pred.size), y = Y_pred, c = 'red', label = 'Predicted', alpha = 0.3)
plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('Bike_Count')
plt.legend()
plt.show()

# plotting a histogram of our residuals
diff = Y_test - Y_pred
plt.hist(x = diff, bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Bike_Count prediction error')
plt.ylabel('Frequency')

# Decision Tree Regressor
# Seperating independent and dependent variables.
x = dataset.iloc[:, [0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[:, 8].values

# SPLITTING THE DATA INTO TRAIN AND TEST.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Regression Model
regressor_DT = DecisionTreeRegressor(max_depth = 2, random_state = 0)
regressor_DT.fit(x_train, y_train)

# Predicting on Test Set
y_pred = regressor_DT.predict(x_test)

# Calculating Mape
MAPE(y_test, y_pred)

# RMSE
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# R Square
# Train Data
y_pred_TR = regressor_DT.predict(x_train) 
r2_score(y_train, y_pred_TR)

# Test Data
r2_score(y_test, y_pred)

# visualising predictions and actual values
plt.scatter(x = range(0, y_test.size), y = y_test, c = 'blue', label = 'Actual', alpha = 0.3)
plt.scatter(x = range(0, y_pred.size), y = y_pred, c = 'red', label = 'Predicted', alpha = 0.3)
plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('Bike_Count')
plt.legend()
plt.show()

# plotting a histogram of our residuals
difference = y_test - y_pred
plt.hist(x = difference, bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Bike_Count prediction error')
plt.ylabel('Frequency')

# Random Forest 
regressor_RF = RandomForestRegressor(max_depth = 2, n_estimators = 10, random_state = 0)
regressor_RF.fit(x_train, y_train)

# Predicting on Test Set
y_pred = regressor_RF.predict(x_test)

# Calculating Mape
MAPE(y_test, y_pred)

# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# R Square
# Train Data
y_pred_TR = regressor_RF.predict(x_train) 
r2_score(y_train, y_pred_TR)

# Test Data
r2_score(Y_test, Y_pred)

# visualising predictions and actual values
plt.scatter(x = range(0, y_test.size), y = y_test, c = 'blue', label = 'Actual', alpha = 0.3)
plt.scatter(x = range(0, y_pred.size), y = y_pred, c = 'red', label = 'Predicted', alpha = 0.3)
plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('Bike_Count')
plt.legend()
plt.show()

# plotting a histogram of our residuals
difference = y_test - y_pred
plt.hist(x = difference, bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Bike_Count prediction error')
plt.ylabel('Frequency')

# Best results of mape and rmse are there in linear model hence i finalise linear
# model for this project.

# Lets train the model on the entire dataset so that it could be used for further
# new test sets.

X_final = dataset.iloc[:, [0,1,2,3,4,5,6,7,9]].values
Y_final = dataset.iloc[:, 8].values

# Creating dummies for categorical variables
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X_final = onehotencoder.fit_transform(X_final).toarray()

# Avoiding dummy variable trap
X_final = X_final[:, [1,3,5,6,7,8,9,10,11,12]]

# Creating intercept for our OLS model
X_final = add_constant(X_final)

# Removing insignificant variables which we learnt from Linear Regression
X_final = X_final[:, [0,1,3,4,5,7,8,9]]

# OLS
regressor_OLS = sm.OLS(endog = Y_final, exog = X_final).fit()
regressor_OLS.summary()

Y_pred_final = regressor_OLS.predict(X_final)

# Calculating Mape(Training Data)
MAPE(Y_final, Y_pred_final)
# MAPE obtained 9.929%, Accuracy = 90.71%

# RMSE(Training Data)
print(np.sqrt(metrics.mean_squared_error(Y_final, Y_pred_final)))

# visualising predictions and actual values
plt.scatter(x = range(0, Y_final.size), y = Y_final, c = 'blue', label = 'Actual', alpha = 0.3)
plt.scatter(x = range(0, Y_pred_final.size), y = Y_pred_final, c = 'red', label = 'Predicted', alpha = 0.3)
plt.title('Actual and predicted values')
plt.xlabel('Observations')
plt.ylabel('Bike_Count')
plt.legend()
plt.show()

# plotting a histogram of our residuals
diff = Y_final - Y_pred_final
plt.hist(x = diff, bins = 40)
plt.title('Histogram of prediction errors')
plt.xlabel('Bike_Count prediction error')
plt.ylabel('Frequency')

#################################### END ########################################
