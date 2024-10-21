import pandas as pd # pandas library help you to stire data as dataframes(2 dimentional tabular structure) used for Data analysis & processing
from sklearn.model_selection import train_test_split # helps to split the data into trainning and testing data
from sklearn.ensemble import RandomForestRegressor # creates differnert decision tress and avgs the results of the trees
from sklearn.metrics import mean_squared_error, r2_score # help to understand how weel our model is trained
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load


# Load the provided CSV file to inspect its contents
file_path = 'RS_Session_259_AU_1203_1.csv'
data = pd.read_csv(file_path)

# Display basic information and first few rows of the dataset
# print(data.info())
# print(data.head())
# print(data.describe())

# Drop the unnamed table in the data-set
data_cleaned = data.drop(columns=["Unnamed: 9"])
# print(data_cleaned.head())

# checking for null values
# print(data_cleaned.isnull().sum())

# Removing rows that have missing values from important features like DO, BOD amd pH
data_cleaned = data_cleaned.dropna(subset=['DO (mg/L)', 'BOD (mg/L)', 'pH'])

# filling FC & FS with median values

# Convert 'FC (MPN/100ml)' and 'FS (MPN/100ml)' to numeric, we are changing all the string, Na to NAN(Not A Number)

data_cleaned['FC (MPN/100ml)'] = pd.to_numeric(data_cleaned['FC (MPN/100ml)'], errors='coerce')
data_cleaned['FS (MPN/100ml)'] = pd.to_numeric(data_cleaned['FS (MPN/100ml)'], errors='coerce')

# we are filling in the median value of FC column in the missing rows 
data_cleaned['FC (MPN/100ml)'] = data_cleaned['FC (MPN/100ml)'].fillna(data_cleaned['FC (MPN/100ml)'].median())

# we are filling in the median value of FS column in the missing rows 
data_cleaned['FS (MPN/100ml)'] = data_cleaned['FS (MPN/100ml)'].fillna(data_cleaned['FS (MPN/100ml)'].median())

# as the year column includes numeric as well as string values we need to convert them to numeric
data_cleaned['Year'] = pd.to_numeric(data_cleaned['Year'], errors='coerce')
# print(data_cleaned.info())
# print(data_cleaned.head())

# we store DO feature in the x and other features in y variables as the DO lvl will help us decide the water quality
X = data_cleaned[['BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)', 'pH']]
y = data_cleaned['DO (mg/L)']

# we split the data into training and testing data 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


# we use Random Tree Forest model as there is no linear relation btw features
model = RandomForestRegressor(n_estimators=100, random_state=3)

# Now we train the model on the training data
model.fit(X_train, y_train)

# making predictions on the test data
y_pred = model.predict(X_test)
# print(y_pred)

# checking the model's performance
mean_sqr_error = mean_squared_error(y_test, y_pred)
r2_sqr_error = r2_score(y_test, y_pred)

# print(mean_sqr_error)
# print(r2)


# Visualize actual vs predicted values for DO (mg/L)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predicted DO')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Perfect Prediction Line')
# plt.xlabel('Actual DO (mg/L)')
# plt.ylabel('Predicted DO (mg/L)')
# plt.title('Actual vs Predicted Dissolved Oxygen (DO) Levels')
# plt.legend()
# plt.grid(True)
# plt.show()

# Feature importance visualization
# feature_importances = model.feature_importances_
# features = X.columns

# plt.figure(figsize=(8, 6))
# sns.barplot(x=feature_importances, y=features, palette="Blues_d")
# plt.title('Feature Importance for DO Prediction')
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.show()

# Save the trained model to a file
model_filename = 'water_quality_model.joblib'
dump(model, model_filename)