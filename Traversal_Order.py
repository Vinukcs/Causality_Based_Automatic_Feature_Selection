from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np

CO2_df = pd.read_csv('CO2_data.csv')
CO2_df = CO2_df.dropna()

X = CO2_df.drop(columns=["CO2"])
y = CO2_df["CO2"]

mutual_info = mutual_info_regression(X, y)

mutual_info_df = pd.DataFrame({"Feature": X.columns, "Mutual_Info": mutual_info})

mutual_info_df = mutual_info_df.sort_values(by="Mutual_Info", ascending=False)

rearranged_CO2_df = CO2_df[mutual_info_df["Feature"].tolist() + ["CO2"]]

rearranged_CO2_df.to_csv('rearranged_CO2_data.csv', index=False)