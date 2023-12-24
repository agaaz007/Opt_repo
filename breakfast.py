from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def compiled_data():
 file_path = 'C:/Users/User/Desktop/Final_Corrected_Book1.xlsx'

 df = pd.read_excel(file_path, skiprows=1)
 df = df.loc[:, ~df.columns.str.contains('Calories')]

 selected_data = df.iloc[[1, 2, 11,12,13, 14, 20, 27, 28,29]]

 new_df = selected_data.copy()

 output_file_path = 'C:/Users/User/Desktop/Book4.xlsx'
 new_df.to_excel(output_file_path, index=False)


 path='C:/Users/User/Desktop/Book4.xlsx'
 df1=pd.read_excel('C:/Users/User/Desktop/Book4.xlsx')
 column_names = df1.columns
 print("Column Names:")
 pattern = re.compile(r'kcal', re.IGNORECASE)
 pattern1 = re.compile(r'Assorted', re.IGNORECASE)

 drop_c=[]
 for column in column_names:
    if re.search(pattern, column) or re.search(pattern1,column):
        # print(f'The column "{column}" contains "kcal".')
        drop_c.append(column)
    # else:
        # print(f'The column "{column}" does not contain "kcal".')
#  print(drop_c)
 df1.drop(columns=drop_c,inplace=True)

 output_file_path = 'C:/Users/User/Desktop/Book5.xlsx'
 df1.to_excel(output_file_path, index=False)
 print(df1.shape)

#  print(new_df)

def sale_data():
 file_path2 = 'C:/Users/User/Desktop/sales_compiled.xlsx'
 df=pd.read_excel(file_path2,skiprows=0)
 print("column names",df.columns)
 df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')  # 'coerce' will handle invalid date formats by setting them to NaT
 start_date = '2023-01-30'
 end_date = '2023-08-30'
 df1 = df.loc[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)]  # Assuming DATE column is in datetime format

# print(df1)

 df_bfast=df1["B.FAST"]
 df_lunch=df1["LUNCH"]
 df_snacks=df1["Snacks"]
 df_dinner=df1["DINNER"]
 return  df_bfast

#bfast, lunch, snacks and dinner data from 30th jan to 31st aug
compiled_data()
file_path3='C:/Users/User/Desktop/Book5.xlsx'
df=pd.read_excel(file_path3)
print(df)
df_snacks_menu=df.iloc[0:2,:]
print(df_snacks_menu)
df_snacks_menu=df_snacks_menu.T
print(df_snacks_menu)
df_snacks_menu= df_snacks_menu.reset_index(drop='True')
print(df_snacks_menu)
# df_snacks_menu= df_snacks_menu.rename(columns = {'0':'Eggs','1':'Parantha'}) 
# print(df_snacks_menu)

# df_snacks_menu=df_snacks_menu.drop('B.FAST')
# df_snacks_menu = df_snacks_menu.str.replace('Main', '')
df_snacks_menu = df_snacks_menu[df_snacks_menu != '']

df_mod=df_snacks_menu #7th aug to 20th aug

snacks_sale=sale_data()
# print(snacks_sale, snacks_sale.isnull().sum().sum())
df_mod = df_mod.reset_index(drop=True)
snacks_sale.reset_index(drop=True, inplace=True)

# print(df_mod, type(df_mod))
df_snacks = pd.concat([df_mod, snacks_sale], axis=1)
print(df_snacks)
# print(df_snacks.columns)
df_snacks.rename(columns={0: 'Eggs'}, inplace=True)
df_snacks.rename(columns={1: 'Parantha'}, inplace=True)
print(df_snacks)
# print(df_snacks["items"].value_counts())
# print("Data Types:")
# print(df_snacks.dtypes)

df_snacks['Eggs'] = pd.Categorical(df_snacks['Eggs'])
df_snacks['Eggs_code'] = df_snacks['Eggs'].cat.codes
df_snacks['Parantha'] = pd.Categorical(df_snacks['Parantha'])
df_snacks['Parantha_code'] = df_snacks['Parantha'].cat.codes
print(df_snacks)
df_snacks=df_snacks.drop(['Eggs'],axis=1)
df_snacks=df_snacks.drop(['Parantha'],axis=1)
print("here", df_snacks)
df_snacks.T.to_excel('C:/Users/User/Desktop/Book6.xlsx')


df_dinner=pd.read_excel('C:/Users/User/Desktop/Book6.xlsx')
df_dinner = df_dinner.iloc[0:3].transpose().dropna()
df_dinner=df_dinner.iloc[1:, :]
print(df_dinner)
df_dinner.columns = ['Dependent_Variable', 'Independent_Variable_1', 'Independent_Variable_2']
X = df_dinner[['Independent_Variable_1', 'Independent_Variable_2']]
print(X)
y = df_dinner['Dependent_Variable']
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(14, 6))

test_predictions = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) for Degree 1 Polynomial:", mse)




    # Scatter plot of actual vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.plot(y_test, y_test, color='red')  # Line of best fit

    # Plotting residuals
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, color='green')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.axhline(y=0, color='red', linestyle='--')

plt.tight_layout()
plt.show()
# print([df_snacks[i].unique().tolist() for i in df_snacks.columns])


# print(df_snacks[1:20])
# print(df_snacks.shape)
print("Check now",df_snacks.isnull().sum())
# Set up the independent variable X and the constant term
# X = sm.add_constant(df_snacks['items_code'])

# # Creating a linear regression model
# model = sm.OLS(df_snacks['Snacks'], X)
# X=list(df['Eggs_code','Parantha_code'])
df_array=df_snacks[['Eggs_code', 'Parantha_code']].to_numpy()
print(df_array)
# X_train, X_test, y_train, y_test = train_test_split(df_array, df_snacks['B.FAST'], test_size=0.3, random_state=42)
# # Set up the independent variable X and the constant term for the training set
# X_train = sm.add_constant(X_train)

# # Creating a linear regression model
# model = sm.OLS(y_train, X_train)
# # Fitting the model
# results = model.fit()
# # Displaying the regression results
# print(results.summary())
# X_train=np.delete(X_train, 0, axis=1)
# y_train=y_train.to_numpy()
# print(X_train)
# print("")
# print(y_train)
# # Plotting the data and the line of best fit
# # plt.figure()
# # plt.scatter(X_train[:,0], y_train, label='Eggs_code', color='blue')
# # plt.xlabel('Eggs_code')
# # plt.ylabel('Breakfast')
# # plt.legend()
# # plt.show()

# # plt.figure()
# # plt.scatter(X_train[:,1], y_train, label='Parantha_code', color='blue')
# # plt.xlabel('Eggs_code')
# # plt.ylabel('Breakfast')
# # plt.legend()
# # plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Scatter plot for multiple linear regression
# fig = plt.figure(figsize=(20,15))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot for 'Eggs_code', 'Parantha_code', and 'B.FAST'
# ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='blue', marker='o')

# # Labeling axes
# ax.set_xlabel('Eggs_code')
# ax.set_ylabel('Parantha_code')
# ax.set_zlabel('Breakfast')

# # Create a meshgrid for the plane of best fit
# xx, yy = np.meshgrid(X_train[:, 0], X_train[:, 1])
# zz = results.params[0] + results.params[1] * xx + results.params[2] * yy

# # Plotting the plane of best fit
# ax.plot_surface(xx, yy, zz, alpha=0.5, color='red')

# plt.show()
# # print(results.predict(X_train))
# # plt.plot(X_train, results.predict(X_train), label='Line of Best Fit', color='red', linewidth=3)
# # plt.xlabel('Item Code')
# # plt.ylabel('Breakfast')
# # plt.legend()

# # Set up the independent variable X and the constant term for the testing set
# X_test = sm.add_constant(X_test)

# # Displaying the regression results for the testing set
# test_results = results.predict(X_test)
# print("Testing set predictions:", test_results)

# # Calculate Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, test_results)
# print("Mean Squared Error (MSE):", mse)
# print(X_train)

# from sklearn.preprocessing import PolynomialFeatures
# import numpy as np
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error
# features = ['Eggs_code','Parantha_code']
# target = 'Predicted_value'

# X = df[features].values.reshape(-1, len(features))
# y = df[target].values
# ols = linear_model.LinearRegression()
# model = ols.fit(X, y)
# model.coef_
# model.intercept_

# model.score(X, y)

# x_pred = np.array([15])
# x_pred = x_pred.reshape(-1, len(features))  # preprocessing required by scikit-learn functions
# model.predict(x_pred)
print(X_test)
y_train=y_train.to_numpy()

# Loop through polynomial degrees from 2 to 11
for degree in range(1, 12):
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    # Transform features for both training and testing sets
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)  # Use the same transformation on the testing set
    print(type(X_test_poly))
    print(type(y_train))
    
    # Fit polynomial regression model
    model = sm.OLS(y_train, X_train_poly)
    results = model.fit()
    
    # Make predictions on the testing set
    test_results_poly = results.predict(X_test_poly)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, test_results_poly)
    
    # Store MSE value in the dictionary
    mse_values[degree] = mse

# Find the degree with the least MSE
best_degree = min(mse_values, key=mse_values.get)




# Display MSE values for different degrees
print("MSE values for different polynomial degrees:")
for degree, mse in mse_values.items():
    print(f"Degree {degree}: MSE = {mse}")

# Display the best degree with the least MSE
print(f"\nThe best polynomial degree is {best_degree} with MSE = {mse_values[best_degree]}")


plt.show()
''' The best method is simple linear regression for this as error is least.
'''