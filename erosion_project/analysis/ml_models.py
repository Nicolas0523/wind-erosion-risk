import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

load_dotenv()

service_account = os.getenv("GEE_SERVICE_ACCOUNT")
key_path = os.getenv("GEE_KEY_PATH")

# Read the file content into a pandas DataFrame
df_erosion = pd.read_csv(r"C:\Users\User\Downloads\wind_erosion_data (2).csv")

df = df_erosion.copy()
def extract_coords(geo_str):
    try:
        geo = json.loads(geo_str)  # читаем JSON
        lon, lat = geo["coordinates"]
        return pd.Series({"longitude": lon, "latitude": lat})
    except Exception:
        return pd.Series({"longitude": None, "latitude": None})

# Применяем функцию
coords = df[".geo"].apply(extract_coords)
df = pd.concat([df, coords], axis=1)
df = df.dropna(subset=["longitude", "latitude"])  # убираем пустые

df["wind_speed"] = abs(df["u_component_of_wind_10m"])
df["Erosionindex"] = (
    df["wind_speed"] * (1-df["NDVI"]) * (1-df["sm_surface"])
    + df["temperature_2m"] * 0.1
    - df["total_precipitation_sum"] * 0.5
    + df["soil_type"] * 0.3
)

features = ["NDVI", "sm_surface", "temperature_2m", "total_precipitation_sum", "soil_type", "wind_speed"]
X = df[features]
y = df["Erosionindex"]

# Corrected the typo 'random_tsate' to 'random_state'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_rf_preds = rf.predict(X_test)

print("Сравнение моделей:\n")
print("Линейная регрессия:")
print("R²:", r2_score(y_test, y_lr_preds))
print("MAE:", mean_absolute_error(y_test, y_lr_preds))
print(df["Erosionindex"].mean())

print("\nRandomForest")
print("R²:", r2_score(y_test, y_rf_preds))
print("MAE:", mean_absolute_error(y_test, y_rf_preds))

print("\nКоэффициенты линейной регрессии:")
# Corrected the typo 'coes_' to 'coef_'
for name, coef in zip(features, lr.coef_):
  print(f"{name}: {coef:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/linear_model.pkl")
joblib.dump(rf, "models/forest_model.pkl")


import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import os

load_dotenv()

service_account = os.getenv("GEE_SERVICE_ACCOUNT")
key_path = os.getenv("GEE_KEY_PATH")

# Read the file content into a pandas DataFrame
df_erosion = pd.read_csv(r"C:\Users\User\Downloads\wind_erosion_data (2).csv")

df = df_erosion.copy()
def extract_coords(geo_str):
    try:
        geo = json.loads(geo_str)  # читаем JSON
        lon, lat = geo["coordinates"]
        return pd.Series({"longitude": lon, "latitude": lat})
    except Exception:
        return pd.Series({"longitude": None, "latitude": None})

# Применяем функцию
coords = df[".geo"].apply(extract_coords)
df = pd.concat([df, coords], axis=1)
df = df.dropna(subset=["longitude", "latitude"])  # убираем пустые

df["wind_speed"] = abs(df["u_component_of_wind_10m"])
df["Erosionindex"] = (
    df["wind_speed"] * (1-df["NDVI"]) * (1-df["sm_surface"])
    + df["temperature_2m"] * 0.1
    - df["total_precipitation_sum"] * 0.5
    + df["soil_type"] * 0.3
)

features = ["NDVI", "sm_surface", "temperature_2m", "total_precipitation_sum", "soil_type", "wind_speed"]
X = df[features]
y = df["Erosionindex"]

# Corrected the typo 'random_tsate' to 'random_state'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_rf_preds = rf.predict(X_test)

print("Сравнение моделей:\n")
print("Линейная регрессия:")
print("R²:", r2_score(y_test, y_lr_preds))
print("MAE:", mean_absolute_error(y_test, y_lr_preds))
print(df["Erosionindex"].mean())

print("\nRandomForest")
print("R²:", r2_score(y_test, y_rf_preds))
print("MAE:", mean_absolute_error(y_test, y_rf_preds))

print("\nКоэффициенты линейной регрессии:")
# Corrected the typo 'coes_' to 'coef_'
for name, coef in zip(features, lr.coef_):
  print(f"{name}: {coef:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/linear_model.pkl")
joblib.dump(rf, "models/forest_model.pkl")
