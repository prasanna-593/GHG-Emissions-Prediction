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
print(df.head(10))                                   
print(df.columns)   
print(df.isnull().sum())                        
