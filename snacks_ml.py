from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split

def compiled_data():
 file_path = '/Users/Agaaz/Downloads/Final_Corrected_Book1.xlsx'

 df = pd.read_excel(file_path, skiprows=1)
 df = df.loc[:, ~df.columns.str.contains('Calories')]

 selected_data = df.iloc[[1, 2, 11,12,13, 14, 20, 27, 28, 29]]


 new_df = selected_data.copy()

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

#  print(new_df)

def sale_data():
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
 return  df_snacks

#bfast, lunch, snacks and dinner data from 30th jan to 31st aug
compiled_data()
file_path3= '/Users/Agaaz/Downloads/11final4.xlsx'
df=pd.read_excel(file_path3)

df_snacks_menu=df.iloc[6,:]
df_snacks_menu = df_snacks_menu.str.replace('Main', '')
df_snacks_menu = df_snacks_menu[df_snacks_menu != '']

df_mod=df_snacks_menu #7th aug to 20th aug

snacks_sale=sale_data()

df_mod = df_mod.reset_index(drop=True)
snacks_sale.reset_index(drop=True, inplace=True)
output_file_path = '/Users/Agaaz/Downloads/output_snacks4.xlsx'
df_mod.to_excel(output_file_path, index=False)
# print(df_mod, type(df_mod))
df_snacks = pd.concat([df_mod, snacks_sale], axis=1)
# print(df_snacks.columns)
df_snacks.rename(columns={6: 'items'}, inplace=True)
# print(df_snacks)
# print(df_snacks["items"].value_counts())
# print("Data Types:")
# print(df_snacks.dtypes)
print(df_snacks['items'].unique() )
print(df_snacks['items'].count() )

df_snacks['items'] = pd.Categorical(df_snacks['items'])
df_snacks['items_code'] = df_snacks['items'].cat.codes
print("sum of item codes",df_snacks["items_code"].nunique())

# print(df_snacks[1:20])
# print(df_snacks.shape)
print("Check now",df_snacks.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(df_snacks[['items_code']], df_snacks['Snacks'], test_size=0.1, random_state=42)

# Create polynomial features for degree 5
poly = PolynomialFeatures(3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Fit polynomial regression model
model = sm.OLS(y_train, X_train_poly).fit()

# Plotting the training data
plt.scatter(X_train, y_train, label='Training Data', color='blue')

# Generating a sequence of item codes for plotting
item_code_range = np.linspace(X_train.min(), X_train.max(), 100)
item_code_range_poly = poly.transform(item_code_range)

# Plotting the polynomial line of best fit
plt.plot(item_code_range, model.predict(item_code_range_poly), label='Polynomial Line of Best Fit - Degree 10', color='red', linewidth=2)

plt.xlabel('Item Code')
plt.ylabel('Snacks')
plt.legend()

# Predictions on the testing set and calculating MSE
test_predictions = model.predict(X_test_poly)
mse = mean_squared_error(y_test, test_predictions)
print("Mean Squared Error (MSE) for Degree 10 Polynomial:", mse)



mse_values = {}

# Loop through polynomial degrees from 2 to 11
for degree in range(1, 12):
    # Create polynomial features
    poly = PolynomialFeatures(degree)
    
    # Transform features for both training and testing sets
    X_train_poly = poly.fit_transform(X_train[['items_code']])
    X_test_poly = poly.transform(X_test[['items_code']])
    
    # Fit polynomial regression model
    model = sm.OLS(y_train, X_train_poly)
    results = model.fit()
    
    # Make predictions on the testing set
    test_results_poly = results.predict(X_test_poly)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, test_results_poly)
    
    # Store MSE value in the dictionary
    mse_values[degree] = mse

best_degree = min(mse_values, key=mse_values.get)

# Display MSE values for different degrees
print("MSE values for different polynomial degrees:")
for degree, mse in mse_values.items():
    print(f"Degree {degree}: MSE = {mse}")

# Display the best degree with the least MSE
print(f"\nThe best polynomial degree is {best_degree} with MSE = {mse_values[best_degree]}")

# Fit the model again with the best degree
poly_best = PolynomialFeatures(best_degree)
X_train_poly_best = poly_best.fit_transform(X_train)
X_test_poly_best = poly_best.transform(X_test)
model_best = sm.OLS(y_train, X_train_poly_best).fit()
test_predictions_best = model_best.predict(X_test_poly_best)

# Calculate residuals for the best degree model
residuals_best = y_test - test_predictions_best

# Plotting residuals for the best degree model
plt.figure(figsize=(10, 6))
plt.scatter(test_predictions_best, residuals_best, color='blue')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted Values for Best Polynomial Degree')
plt.show()