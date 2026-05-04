import pandas as pd

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

columns = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age",
    "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker", "target"
]

df = pd.read_csv(url, sep=" ", header=None, names=columns)

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["target"] = df["target"].map({1: 1, 2: 0})

df_encoded = pd.get_dummies(df, drop_first=True)

print(df_encoded.head())
print(df_encoded.shape)
