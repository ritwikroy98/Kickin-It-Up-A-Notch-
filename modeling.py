# Reading the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = pd.read_csv("C:\\Users\\thepo\\Documents\\GitHub\\MSIS - Spring\\Kickin-It-Up-A-Notch-\\Data\\cleaned_data.csv")
df = df.drop(df.columns[[0, 1]], axis=1)    ### Dropping ID (blank) and Index (saved) column
df

df.plot.scatter(x='retailPrice', y='averageDeadstockPrice')
plt.show()    ####Linear but doesnt really go up or down

df.plot.scatter(x='lowestAsk', y='averageDeadstockPrice')
plt.show()    ####Linear

df.plot.scatter(x='numberOfAsks', y='averageDeadstockPrice')
plt.show()    #### Clustered around a tight space

df.plot.scatter(x='salesThisPeriod', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='salesThisPeriod', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='salesLastPeriod', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='highestBid', y='averageDeadstockPrice')
plt.show()    #### Linear

df.plot.scatter(x='numberOfBids', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='annualHigh', y='averageDeadstockPrice')
plt.show()    #### Linear

df.plot.scatter(x='annualLow', y='averageDeadstockPrice')
plt.show()    #### Linear

df.plot.scatter(x='volatility', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='deadstockSold', y='averageDeadstockPrice')
plt.show()    #### Straight Line

df.plot.scatter(x='pricePremium', y='averageDeadstockPrice')
plt.show()    #### Linear

df.plot.scatter(x='lastSale', y='averageDeadstockPrice')
plt.show()    #### Linear

df['brand'] = df['brand'].astype("factor")
price_map = {'adidas': 1, 'Nike': 2, 'New Balance': 3, 'Jordan': 4}
df['brand_dummy'] = df['brand'].map(price_map)
df['brand_dummy'] = df['brand_dummy'].astype("int")

## Linear Reg
#### intent1_01 ~ attitude1_01 + peruse01 + satis01 + satis04 + peruse04
linreg1 = LinearRegression(fit_intercept=True)
linreg1.fit(df[['retailPrice','brand_dummy','lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']], df.averageDeadstockPrice)
linreg1.summary()

## Regression
import statsmodels.formula.api as smf
linreg2 = smf.ols('averageDeadstockPrice ~ retailPrice + brand_dummy + lowestAsk + numberOfAsks + salesThisPeriod + highestBid + numberOfBids + annualHigh + annualLow + volatility + deadstockSold + pricePremium + lastSale', df).fit()
linreg2.summary()

### Calculating VIF
linreg1.fit(df[['lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']], df.retailPrice)
vif1 = 1/(1 - linreg1.score(df[['lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']], df.retailPrice))

### Calculating VIF
linreg1.fit(df[['lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']], df.retailPrice)
vif1 = 1/(1 - linreg1.score(df[['lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']], df.retailPrice))

## Try - 1 all variables
X = df[['retailPrice','lowestAsk','numberOfAsks','salesThisPeriod','highestBid','numberOfBids','annualHigh','annualLow','volatility','deadstockSold','pricePremium','lastSale']]

# add constant column for intercept
X = sm.add_constant(X)

# calculate VIF for each predictor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["predictor"] = X.columns

# display results
print(vif)
#### Removing all price variables.

## Try - 2 all variables except price
Y = df[['retailPrice','lowestAsk','numberOfAsks','salesThisPeriod','numberOfBids','volatility','deadstockSold','pricePremium']]

# add constant column for intercept
Y = sm.add_constant(Y)

# calculate VIF for each predictor
vif2 = pd.DataFrame()
vif2["VIF Factor"] = [variance_inflation_factor(Y.values, i) for i in range(Y.shape[1])]
vif2["predictor"] = Y.columns

# display results
print(vif2)

# Regression Model - 2
linreg2 = smf.ols('averageDeadstockPrice ~ retailPrice + brand_dummy + lowestAsk + numberOfAsks + salesThisPeriod + numberOfBids + volatility + deadstockSold + pricePremium', df).fit()
linreg2.summary()

# Correlation Matrix
df_corr = df.['']
corr_matrix = df.corr()

# Display the correlation matrix
print(corr_matrix)
corr_matrix.to_csv('Data\corr_matrix.csv', sep=',')

# Regression for all but yellow (Refer to cleaned_data csv for coloring)
## Get dummies
dummy_df = pd.get_dummies(df['brand'], drop_first=True)
dummy_df.dtypes

# Assigning the resulting dataframe to a new variable
concat_data = pd.concat([df, dummy_df], axis=1)

# Dropping the original column since it's no longer needed
concat_data = concat_data.drop(concat_data[['brand', 'brand_dummy']], axis=1) 
concat_data = concat_data.drop(concat_data[['brand_dummy']], axis=1)
# Viewing the resulting dataframe with the dummy variables
concat_data.columns = [col.replace(' ', '_') for col in concat_data.columns]
print(concat_data.head())

# Model
linreg3 = smf.ols('averageDeadstockPrice ~ belowRetail + retailPrice + lowestAskSize + salesLastPeriod + highestBid + highestBidSize + numberOfBids + volatility + deadstockSold + changePercentage + absChangePercentage + deadstockSoldRank + pricePremiumRank + averageDeadstockPriceRank + New_Balance + Nike + adidas', concat_data).fit()
linreg3.summary()

# Multicolinearity
## Try - 2 all variables except price
concat_data.dtypes
df['New_Balance'] = df['New_Balance'].astype("int")
df['Nike'] = df['Nike'].astype("int")
df['adidas'] = df['adidas'].astype("int")
concat_data.isna().sum()
price_map = {'True': 1, 'False': 2}
concat_data['belowRetail']
concat_data['belowRetail_dummy'] = concat_data['belowRetail'].map(price_map)
concat_data['belowRetail_dummy']
concat_data['belowRetail_dummy'] = concat_data['belowRetail_dummy'].astype("int")

Z = concat_data[['retailPrice','salesLastPeriod','highestBid','numberOfBids','volatility', 'deadstockSold', 'changePercentage', 'absChangePercentage', 'deadstockSoldRank', 'pricePremiumRank', 'averageDeadstockPriceRank', 'New_Balance', 'Nike', 'adidas']]

# add constant column for intercept
Z = sm.add_constant(Z)
vif3 = pd.DataFrame()
vif3["VIF Factor"] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]
vif3["predictor"] = Z.columns

# display results
print(vif3)

# Train and Test Split - sklearn function
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(concat_data[['belowRetail','retailPrice','salesLastPeriod','highestBid','numberOfBids','volatility', 'deadstockSold', 'changePercentage', 'absChangePercentage', 'deadstockSoldRank', 'pricePremiumRank', 'averageDeadstockPriceRank', 'New_Balance', 'Nike', 'adidas']], concat_data['averageDeadstockPrice'], test_size=0.3, random_state=12345)

# Initialize and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
score_1 = model.score(X_train, y_train)
print('R^2 score:', score_1)
y_pred = model.predict(X_test)
## Evaluate
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

## Try 4 (Removing unnecessary variables)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(concat_data[['highestBid', 'New_Balance', 'Nike', 'adidas']], concat_data['averageDeadstockPrice'], test_size=0.3, random_state=12345)

# Initialize and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
score_1 = model.score(X_train, y_train)
print('R^2 score:', score_1)
y_pred = model.predict(X_test)
## Evaluate
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

## DECISION TREE REGRESSOR
# Splitting the data into training and testing sets
from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(concat_data[['highestBid', 'New_Balance', 'Nike', 'adidas']], concat_data['averageDeadstockPrice'], test_size=0.3, random_state=12345)

# Training the decision tree regression model
from sklearn import tree
regressor = DecisionTreeRegressor(max_depth=2)
regressor.fit(X_train, y_train)

data_sub = concat_data[['belowRetail','retailPrice','salesLastPeriod','highestBid','numberOfBids','volatility', 'deadstockSold', 'changePercentage', 'absChangePercentage', 'deadstockSoldRank', 'pricePremiumRank', 'averageDeadstockPriceRank', 'New_Balance', 'Nike', 'adidas']]
col_names = list(concat_data.columns.values)
tre1 = tree.DecisionTreeRegressor(min_samples_split=2,min_samples_leaf=5, max_depth=2).fit(data_sub, concat_data.averageDeadstockPrice)
tree.plot_tree(tre1, feature_names=col_names,filled=True,rounded=True)
plt.show()
plt.figure(figsize=(10,8))
tree.plot_tree(tre1,feature_names=col_names,
                filled=True,rounded=True,
                fontsize=10)
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2): ", r2)

# Predicting the output for test set
y_pred = regressor.predict(X_test)

# Plotting the results
plt.figure()
plt.scatter(X_train, y_train, s=20, edgecolor="black", c="darkorange", label="training data")
plt.scatter(X_test, y_test, s=20, edgecolor="black", c="green", label="testing data")
plt.plot(X_test, y_pred, color="cornflowerblue", label="prediction")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

