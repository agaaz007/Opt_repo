import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Load your dataset
file_path3 = '/Users/Agaaz/Downloads/16final4.xlsx'
df = pd.read_excel(file_path3)
df = df.iloc[0:3].transpose().dropna()
df.columns = ['Dependent_Variable', 'Independent_Variable_1', 'Independent_Variable_2']


# Assuming 'Dependent_Variable', 'Independent_Variable_1', 'Independent_Variable_2' are your column names
X = df[['Independent_Variable_1', 'Independent_Variable_2']]
y = df['Dependent_Variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Instantiate and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for Random Forest:", mse)

# Optional: Additional code for plotting or further analysis
# Assuming your Random Forest model is named 'model'
n_trees = 3  # Number of trees you want to plot
fig, axes = plt.subplots(nrows=1, ncols=n_trees, figsize=(20, 4))

for i in range(n_trees):
    tree = model.estimators_[i]
    plot_tree(tree, filled=True, ax=axes[i], feature_names=X.columns.tolist(), max_depth=3)

    axes[i].set_title(f'Tree {i+1}')

plt.tight_layout()
plt.show()


def compute_cost(x_train, y_train, w, b):

    '''
    x_train, y_train : training dataset
    w, b : our model

    returns: cost function (how bad is our hypothesis)
    '''
    # initialization
    m = len(x_train)
    J = 0

    for i in range(m):
      x_i = x_train[i] # getting the data points
      y_i = y_train[i]

      y_predicted = w * x_i + b # calculating the prediction
      loss = (y_predicted - y_i)**2 # calculating the loss
      J = J + loss

    J = J / (2 * m)
    return J

def compute_gradient(x_train, y_train, w, b):
  m = len(x_train) # initialize variables
  dj_dw = 0
  dj_db = 0

  for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    y_predicted = w * x_i + b

    dj_dw_i = (y_predicted - y_i)*x_i 
    dj_db_i = (y_predicted - y_i)

    dj_dw += dj_dw_i 
    dj_db += dj_db_i

  dj_dw = dj_dw / m 
  dj_db = dj_db / m

  return dj_dw, dj_db