# Climate & Population Forecasting using Linear Regression 
 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_absolute_error 
 
# 1. LOAD CLEANED DATASETS 
 
climate_df = pd.read_csv("cleaned_climate_dataset.csv")     # Climate data 
population_df = pd.read_csv("cleaned_population_dataset.csv")  # Population 
data 
 
# 2. CLIMATE PREDICTION MODEL 
 
print("\n=== CLIMATE PREDICTION MODEL ===") 
 
# Features and target 
X_climate = climate_df[['CO2_ppm', 'SeaLevel_mm']] 
y_climate = climate_df['Temperature_C'] 
 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X_climate, y_climate, 
test_size=0.2, random_state=42) 
 
# Model training 
climate_model = LinearRegression() 
climate_model.fit(X_train, y_train) 
 
# Predictions 
y_pred = climate_model.predict(X_test) 
 
# Evaluation 
print("R² Score (Climate):", round(r2_score(y_test, y_pred) * 100, 2), "%") 
print("MAE (Climate):", round(mean_absolute_error(y_test, y_pred), 3)) 
 
# Visualization 
plt.scatter(y_test, y_pred, color="blue") 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--") 
plt.xlabel("Actual Temperature (°C)") 
plt.ylabel("Predicted Temperature (°C)") 
plt.title("Climate Model: Actual vs Predicted") 
plt.show() 
 
# Future Predictions  
future_climate = pd.DataFrame({ 
    "CO2_ppm": [550, 510, 565], 
    "SeaLevel_mm": [350, 430, 420] 
}, index=["2050", "2060", "2070"]) 
 
future_climate_pred = climate_model.predict(future_climate) 
print("\nFuture Climate Predictions (°C):") 
print(future_climate_pred) 
 
# CLIMATE TREND VISUALIZATION 
# Convert year column to numeric (in case it's string) 
climate_df['Year'] = pd.to_numeric(climate_df['Year'], errors='coerce') 
 
plt.figure(figsize=(8,5)) 
 
# Plot historical trend 
plt.plot(climate_df['Year'], climate_df['Temperature_C'], label="Historical 
Temp", color="blue") 
 
# Plot predictions (use numeric years) 
future_years = [2050,2060,2070] 
plt.scatter(future_years, future_climate_pred, color="red", marker="x", 
s=100, label="Predicted Temp") 
 
plt.xlabel("Year") 
plt.ylabel("Temperature (°C)") 
plt.title("Climate Trend: Historical vs Predicted") 
plt.legend() 
plt.grid(True) 
plt.show() 
 
#INSIGHTS 
print(f"Climate: Based on trends, the model suggests that by 2050, " 
      f"global temperature could rise to approx {round(future_climate_pred[
1],2)}°C " 
      f"if CO₂ and sea levels continue to rise.") 
 
# 3. POPULATION PREDICTION MODEL 
 
print("\n=== POPULATION PREDICTION MODEL ===") 
 
# Features and target 
X_pop = population_df[['BirthRate', 'DeathRate', 'LifeExpectancy', 
'UrbanPopulation%']] 
y_pop = population_df['Population'] 
 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X_pop, y_pop, test_size=0.2, 
random_state=42) 
 
# Train the Linear Regression model 
population_model = LinearRegression() 
population_model.fit(X_train, y_train) 
 
# Predictions 
y_pred = population_model.predict(X_test) 
 
# Evaluation 
print("R² Score (Population):", round(r2_score(y_test, y_pred) * 100, 2), 
"%") 
print("MAE (Population):", round(mean_absolute_error(y_test, y_pred), 2)) 
 
# Visualization 
plt.scatter(y_test, y_pred, color="green") 
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--") 
plt.xlabel("Actual Population") 
plt.ylabel("Predicted Population") 
plt.title("Population Model: Actual vs Predicted") 
plt.show() 
 
# Future Predictions  
future_population = pd.DataFrame({ 
    "BirthRate": [16, 15, 19], 
    "DeathRate": [5, 8, 4], 
    "LifeExpectancy": [78, 75, 85], 
    "UrbanPopulation%": [75,82,72] 
}, index=["2050", "2060", "2070"]) 
 
future_population_pred = population_model.predict(future_population) 
print("\nFuture Population Predictions:") 
print(future_population_pred) 
 
#CLIMATE TREND VISUALIZATION 
# Convert year column to numeric (in case it's string) 
population_df['Year'] = pd.to_numeric(population_df['Year'], 
errors='coerce') 
 
plt.figure(figsize=(8,5)) 
 
# Plot historical population trend 
plt.plot(population_df['Year'], population_df['Population'], label="Historical 
Population", color="green") 
 
# Plot predictions (numeric years) 
future_years = [2050,2060,2070] 
plt.scatter(future_years, future_population_pred, color="red", marker="x", 
s=100, label="Predicted Population") 
 
plt.xlabel("Year") 
plt.ylabel("Population") 
plt.title("Population Trend: Historical vs Predicted") 
plt.legend() 
plt.grid(True) 
plt.show() 
 
#INSIGHTS 
print(f" Population: The model projects that global population could reach 
nearly " 
      f"{int(future_population_pred[-1])} by 2050, driven by declining death 
rates " 
      f"and higher urbanization.") 
 
print ("\n Key Takeaway: These insights highlight the strong connection 
between " 
      "Human growth and climate stress, underlining the need for sustainable 
development policies.")