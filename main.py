from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv('wells_info_with_prod.csv')
# print(df.head(5))
selected_features = pd.DataFrame()
selected_features["API"] = df["API"]
selected_features["PermitDate"] = df["PermitDate"]
selected_features["BasinName"] = df["BasinName"]
selected_features["LonWGS84"] = df["LonWGS84"]
selected_features["Prod1Year"] = df["Prod1Year"]

selected_features['PermitDate:year'] = pd.to_datetime(df['PermitDate']).dt.year.astype(float)
selected_features['PermitDate:month'] = pd.to_datetime(df['PermitDate']).dt.month.astype(float)
selected_features['PermitDate:day'] = pd.to_datetime(df['PermitDate']).dt.day.astype(float)
selected_features = selected_features.drop('PermitDate', axis=1)

# Создание dummy-переменных для столбца "BasinName"
dummy_variables = pd.get_dummies(selected_features["BasinName"], prefix="Basin")
selected_features = selected_features.drop("BasinName", axis=1)
selected_features = pd.concat([selected_features, dummy_variables], axis=1)
selected_features = selected_features.astype(float)

# print(selected_features.head(5), "\n")

features = selected_features.drop("Prod1Year", axis=1)
target_var = selected_features["Prod1Year"]

# Разделение данных на обучающий и тестовый наборы
features_train, features_test, target_var_train, target_var_test = train_test_split(features, target_var, test_size=0.2, random_state=42)

# print("Train set:\n", features_train.head(5), "\n")
# print("Test set:\n", features_test.head(5))

scaler = StandardScaler()

# Масштабирование обучающего набора данных
features_train_scaled = scaler.fit_transform(features_train)
target_var_scaled = scaler.fit_transform(target_var_train.values.reshape(-1, 1))

# Масштабирование тестового набора данных
features_test_scaled = scaler.transform(features_test.values.reshape(-1, 1))
target_test_scaled = scaler.transform(target_var_test.values.reshape(-1, 1))

# Вывод первых пяти строк масштабированного обучающего набора данных
print(features_train_scaled[:5])

# print(features_train_scaled.mean())
# print(features_train_scaled.var())
