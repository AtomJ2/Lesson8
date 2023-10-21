from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("wells_info_with_prod.csv")
# print(df.head(5))
selected_features = df[["API", "PermitDate", "SpudDate", "CompletionDate", "formation", "BasinName", "StateName", "LatWGS84", "LonWGS84", "Prod1Year"]]
print(selected_features.head(5), "\n")

# Разделение данных на признаки (X) и целевую переменную (y)
features = selected_features.drop("Prod1Year", axis=1)
target_var = selected_features["Prod1Year"]

# Разделение данных на обучающий и тестовый наборы
features_train, features_test, target_var_train, target_var_test = train_test_split(features, target_var, test_size=0.2, random_state=42)

# Вывод размеров обучающего и тестового наборов
print("Train set:\n", features_train.head(5), "\n")
print("Test set:\n", features_test.head(5))

# Создание объекта StandardScaler
scaler = StandardScaler()

# Масштабирование обучающего набора данных
features_train_scaled = scaler.fit_transform(features_train)
target_var_scaled = scaler.fit_transform(target_var_train)

# Масштабирование тестового набора данных
features_test_scaled = scaler.transform(features_test)
target_test_scaled = scaler.transform(target_var_test)

# Вывод первых пяти строк масштабированного обучающего набора данных
print(features_train_scaled[:5])
