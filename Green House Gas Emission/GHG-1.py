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
df.drop(columns=['Name','Code','Year'], inplace=True)
print("After dropping columns, here’s the first row:")
print(df.head(1))

df.shape
df.columns
X = df.drop(columns=['Supply Chain Emission Factors with Margins']) # Feature set excluding the target variable
y = df['Supply Chain Emission Factors with Margins'] # Target variable 

X.head()
y.head()
# Count plot for Substance
plt.figure(figsize=(6, 3))
sns.countplot(x=df["Substance"])
plt.title("Count Plot: Substance")
plt.xticks()
plt.tight_layout()
plt.show()

# Count plot for Unit
plt.figure(figsize=(6, 3))
sns.countplot(x=df["Unit"])
plt.title("Count Plot: Unit")
plt.tight_layout()
plt.show()
# Count plot for Source
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Source"])
plt.title("Count Plot: Source (Industry vs Commodity)")
plt.tight_layout()
plt.show()

df.columns
df.select_dtypes(include=np.number).corr() # Checking correlation between numerical features 
df.info() # Checking data types and non-null counts after mapping 
# Correlation matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()