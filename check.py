import pandas as pd

file_path = 'C:/Users/Ajay_XPS_9380/desktop/discharge_with_social_final.csv'

df = pd.read_csv(file_path)

df_2 = df.drop_duplicates()

print(f"{len(df)} vs {len(df_2)}")