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
 return  df_lunch

#bfast, lunch, snacks and dinner data from 30th jan to 31st aug
compiled_data()
file_path3='C:/Users/User/Desktop/Book5.xlsx'
df=pd.read_excel(file_path3)
print(df)
df_snacks_menu=df.iloc[2:6,:]
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
df_snacks.rename(columns={2: '1'}, inplace=True)
df_snacks.rename(columns={3: '2'}, inplace=True)
df_snacks.rename(columns={4: '3'}, inplace=True)
df_snacks.rename(columns={5: '4'}, inplace=True)

print(df_snacks)
# print(df_snacks["items"].value_counts())
# print("Data Types:")
# print(df_snacks.dtypes)

df_snacks['1'] = pd.Categorical(df_snacks['1'])
df_snacks['1_code'] = df_snacks['1'].cat.codes
df_snacks['2'] = pd.Categorical(df_snacks['2'])
df_snacks['2_code'] = df_snacks['2'].cat.codes
df_snacks['3'] = pd.Categorical(df_snacks['3'])
df_snacks['3_code'] = df_snacks['3'].cat.codes
df_snacks['4'] = pd.Categorical(df_snacks['4'])
df_snacks['4_code'] = df_snacks['4'].cat.codes
print(df_snacks)

# Drop unnecessary columns
df_snacks = df_snacks.drop(['1', '2','3','4'], axis=1)

# Save to Book6.xlsx
df_snacks.T.to_excel('C:/Users/User/Desktop/Book7.xlsx')

# Read data from Book6.xlsx
df_dinner = pd.read_excel('C:/Users/User/Desktop/Book7.xlsx')
df_dinner = df_dinner.iloc[0:5].transpose().dropna()
df_dinner = df_dinner.iloc[1:, :]
print(df_dinner)
df_dinner.columns = ['Dependent_Variable', 'Independent_Variable_1', 'Independent_Variable_2', 'Independent_Variable_3', 'Independent_Variable_4']
X = df_dinner[['Independent_Variable_1', 'Independent_Variable_2', 'Independent_Variable_3', 'Independent_Variable_4']]
y = df_dinner['Dependent_Variable']

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
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
print([df_snacks_menu[i].unique().tolist() for i in df_snacks_menu.columns])

mse_values={}
y_train = y_train.to_numpy()
# Loop through polynomial degrees from 2 to 11
for degree in range(1, 12):
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    # Transform features for both training and testing sets
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)  # Use the same transformation on the testing set
    
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

