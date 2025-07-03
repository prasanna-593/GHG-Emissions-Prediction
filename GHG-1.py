
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Script is running successfully!")

# Load Excel
excel_file = 'SupplyChainEmissionFactorsforUSIndustriesCommodities.xlsx'  # Make sure file is in same folder
years = range(2010, 2017)
df_1 = pd.read_excel(excel_file, sheet_name=f'{years[0]}_Detail_Commodity')

# Show first 5 rows
print(df_1.head())

df_2 = pd.read_excel(excel_file, sheet_name=f'{years[0]}_Detail_Industry')
print(df_2.head())

all_data = []

for year in years:
    try:
        df_com = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Commodity')
        df_ind = pd.read_excel(excel_file, sheet_name=f'{year}_Detail_Industry')
        
        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'
        df_com['Year'] = df_ind['Year'] = year
        
        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()

        df_com.rename(columns={
            'Commodity Code': 'Code',
            'Commodity Name': 'Name'
        }, inplace=True)
        
        df_ind.rename(columns={
            'Industry Code': 'Code',
            'Industry Name': 'Name'
        }, inplace=True)
        
        all_data.append(pd.concat([df_com, df_ind], ignore_index=True))
        
    except Exception as e:
        print(f"Error processing year {year}: {e}")
df = pd.concat(all_data, ignore_index=True)
df.drop(columns=['Unnamed: 7'], inplace=True, errors='ignore')

# Check columns and summary
print("\nDataFrame Info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe().T)
print("\nNull values in each column:")
print(df.isnull().sum())

# Visualize distribution
print("\nAbout to show histogram of target variable...")
sns.histplot(df['Supply Chain Emission Factors with Margins'].dropna(), bins=50, kde=True)
plt.title('Target Variable Distribution')
plt.xlabel('Supply Chain Emission Factors with Margins')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()
print("Plot shown successfully.\n")

# Check categorical variables
print("All Columns:", df.columns.tolist())

print("\nSubstance value counts:")
print(df['Substance'].value_counts())

print("\nUnit value counts:")
print(df['Unit'].value_counts())

print("\nUnique Unit values:")
print(df['Unit'].unique())

print("\nSource value counts:")
print(df['Source'].value_counts())

print("\nUnique Substance values:")
print(df['Substance'].unique())

# Mapping categorical values to integers
substance_map = {'carbon dioxide': 0, 'methane': 1, 'nitrous oxide': 2, 'other GHGs': 3}
unit_map = {'kg/2018 USD, purchaser price': 0, 'kg CO2e/2018 USD, purchaser price': 1}
source_map = {'Commodity': 0, 'Industry': 1}

df['Substance'] = df['Substance'].map(substance_map)
df['Unit'] = df['Unit'].map(unit_map)
df['Source'] = df['Source'].map(source_map)

print("\nAfter mapping:")
print("Substance unique values:", df['Substance'].unique())
print("Unit unique values:", df['Unit'].unique())
print("Source unique values:", df['Source'].unique())

print("\nData types after mapping:")
print(df.info())

# Unique Codes and Names
print("\nUnique values in Code:")
print(df['Code'].unique())

print("\nUnique values in Name:")
print(df['Name'].unique())

print("\nNumber of unique Names:", len(df['Name'].unique()))

# Top 10 emitters
top_emitters = df[['Name', 'Supply Chain Emission Factors with Margins']] \
    .groupby('Name').mean().sort_values(
    'Supply Chain Emission Factors with Margins', ascending=False).head(10).reset_index()

print("\nTop 10 Emitters by average emission factor:")
print(top_emitters)

# Plot Top Emitters
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Supply Chain Emission Factors with Margins',
    y='Name',
    data=top_emitters,
    hue='Name',
    palette='viridis',
    legend=False
)

# Add ranking labels
for i, (value, name) in enumerate(zip(top_emitters['Supply Chain Emission Factors with Margins'], top_emitters.index), start=1):
    plt.text(value + 0.01, i - 1, f'#{i}', va='center', fontsize=11, fontweight='bold', color='black')

plt.title('Top 10 Emitting Industries', fontsize=14, fontweight='bold')
plt.xlabel('Emission Factor (kg CO2e/unit)')
plt.ylabel('Industry')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ✅ FIXED: Define X and y before scaling
df.drop(columns=['Name', 'Code', 'Year'], inplace=True)

X = df.drop(columns=['Supply Chain Emission Factors with Margins'])  # features
y = df['Supply Chain Emission Factors with Margins']  # target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for viewing
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Show result
print("Scaled Features (first 5 rows):")
print(X_scaled_df.head())

X_scaled[0].min(),X_scaled[0].max()
np.round(X_scaled.mean()),np.round(X_scaled.std())
X.shape
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Splitting data into training and testing sets 
X_train.shape
X_test.shape
RF_model = RandomForestRegressor(random_state=42) # Initializing Random Forest Regressor 
RF_model.fit(X_train, y_train) # Fitting the model on training data 
RF_y_pred = RF_model.predict(X_test) # Making predictions on the test set 

RF_y_pred[:20]
RF_mse = mean_squared_error(y_test, RF_y_pred) # Calculating Mean Squared Error (MSE)
RF_rmse = np.sqrt(RF_mse) # Calculating Root Mean Squared Error (RMSE)
# Calculating R² score
RF_r2 = r2_score(y_test, RF_y_pred)

print(f'RMSE: {RF_rmse}')
print(f'R² Score: {RF_r2}')
from sklearn.linear_model import LinearRegression # Importing Linear Regression model 
LR_model = LinearRegression() # Initializing Linear Regression model
# Fitting the Linear Regression model on training data

LR_model.fit(X_train, y_train)

LR_y_pred = LR_model.predict(X_test) # Making predictions on the test set using Linear Regression model 


LR_mse = mean_squared_error(y_test, LR_y_pred) # Calculating Mean Squared Error (MSE) for Linear Regression model
LR_rmse = np.sqrt(LR_mse) # Calculating Root Mean Squared Error (RMSE) for Linear Regression model 
LR_r2 = r2_score(y_test, LR_y_pred) # Calculating R² score for Linear Regression model 

print(f'RMSE: {LR_rmse}')
print(f'R² Score: {LR_r2}')
# Hyperparameter tuning for Random Forest Regressor using GridSearchCV 
# Define the parameter grid for hyperparameter tuning 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Perform grid search with cross-validation to find the best hyperparameters 
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)

# Fit the grid search model on the training data 
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
# Use the best model to make predictions on the test set 
y_pred_best = best_model.predict(X_test)


HP_mse = mean_squared_error(y_test, y_pred_best)
HP_rmse = np.sqrt(HP_mse)
HP_r2 = r2_score(y_test, y_pred_best)

print(f'RMSE: {HP_rmse}')
print(f'R² Score: {HP_r2}')
# Create a comparative DataFrame for all models
results = {
    'Model': ['Random Forest (Default)', 'Linear Regression', 'Random Forest (Tuned)'],
    'MSE': [RF_mse, LR_mse, HP_mse],
    'RMSE': [RF_rmse, LR_rmse, HP_rmse],
    'R2': [RF_r2, LR_r2, HP_r2]
}

# Create a DataFrame to compare the results of different models
comparison_df = pd.DataFrame(results)
print(comparison_df)
