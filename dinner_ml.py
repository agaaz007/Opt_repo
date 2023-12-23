from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

def compiled_df_dinner():
 file_path = '/Users/Agaaz/Downloads/Final_Corrected_Book1.xlsx'

 df = pd.read_excel(file_path, skiprows=1)
 df = df.loc[:, ~df.columns.str.contains('Calories')]

 selected_df_dinner = df.iloc[[1, 2, 11,12,13, 14, 20, 27, 28, 29]]


 new_df = selected_df_dinner.copy()

 output_file_path = '/Users/Agaaz/Downloads/0final4.xlsx'
 new_df.to_excel(output_file_path, index=False)


 path='/Users/Agaaz/Downloads/0final4.xlsx'
 df1=pd.read_excel('/Users/Agaaz/Downloads/0final4.xlsx')
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

 output_file_path = '/Users/Agaaz/Downloads/11final4.xlsx'
 df1.to_excel(output_file_path, index=False)
 print(df1.shape)


def sale_df_dinner():
 file_path2 = '/Users/Agaaz/Downloads/sales_compiled_final.xlsx'
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
 return  df_dinner


compiled_df_dinner()
file_path3= '/Users/Agaaz/Downloads/11final4.xlsx'
df=pd.read_excel(file_path3)

df_dinner_menu=df.iloc[7:,:]
df_dinner_menu = df_dinner_menu[df_dinner_menu != '']
df_mod=df_dinner_menu #30th Jan to 30th aug

dinner_sale=sale_df_dinner()
df_mod = df_mod.reset_index(drop=True)
print(df_mod)
path='/Users/Agaaz/Downloads/15final4.xlsx'
df_mod.to_excel(path,index=False)
dinner_sale.reset_index(drop=True, inplace=True)
print(dinner_sale, type(dinner_sale),dinner_sale.shape)

# df_mod.apply(LabelEncoder().fit_transform)
# print(df_mod.iloc[:, :5])
# print(df_lunch)
# df_lunch.rename(columns={6: 'items'}, inplace=True)


path='/Users/Agaaz/Downloads/16final4.xlsx'
df_dinner=pd.read_excel(path)
df_dinner = df_dinner.iloc[0:3].transpose().dropna()
df_dinner.columns = ['Dependent_Variable', 'Independent_Variable_1', 'Independent_Variable_2']
X = df_dinner[['Independent_Variable_1', 'Independent_Variable_2']]
y = df_dinner['Dependent_Variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# plt.figure(figsize=(14, 6))

# test_predictions = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error (MSE) for Degree 1 Polynomial:", mse)




#     # Scatter plot of actual vs predicted values
# plt.subplot(1, 2, 1)
# plt.scatter(y_test, y_pred, color='blue')
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Actual vs Predicted Values')
# plt.plot(y_test, y_test, color='red')  # Line of best fit

#     # Plotting residuals
# plt.subplot(1, 2, 2)
# residuals = y_test - y_pred
# plt.scatter(y_pred, residuals, color='green')
# plt.xlabel('Predicted')
# plt.ylabel('Residuals')
# plt.title('Residuals vs Predicted Values')
# plt.axhline(y=0, color='red', linestyle='--')


degree = 2  # You can change the degree as needed
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Create and fit the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) for Degree", degree, "Polynomial:", mse)

# Plotting
plt.figure(figsize=(14, 6))

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



best_degree = 1
min_mse = float('inf')

# Iterate through degrees 1 to 11
for degree in range(1, 12):
    # Create a polynomial regression model for this degree
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, y_train)

    # Predict using the model
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE) for Degree {degree} Polynomial:", mse)

    # Update the best degree and min_mse if this model is better
    if mse < min_mse:
        min_mse = mse
        best_degree = degree

print(f"Best Degree: {best_degree} with MSE: {min_mse}")








plt.tight_layout()
plt.show()