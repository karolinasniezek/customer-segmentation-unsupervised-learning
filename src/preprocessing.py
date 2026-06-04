import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

np.random.seed(42)

df = pd.read_csv('../data/marketing_campaign.csv')
print(df.columns)

missing_values = df.isnull().sum()
print(missing_values)

label_encoder = LabelEncoder()
df_categoricals = df.select_dtypes(include=["object"])
df_encoded_categoricals = df_categoricals.apply(label_encoder.fit_transform)
df[df_encoded_categoricals.columns] = df_encoded_categoricals
print(df[df_encoded_categoricals.columns])

df_numerical = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=df_numerical.columns)

df[df_numerical.columns] = df_scaled

