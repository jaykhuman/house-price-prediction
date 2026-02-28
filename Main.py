# ============================================================
#   HOUSE PRICE PREDICTION - Multiple ML Models Comparison
#   Author: [Your Name]
#   Description: Predict house prices using multiple ML models
#                and compare their performance.
# ============================================================

# ── 1. IMPORT LIBRARIES ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ── 2. LOAD DATASET ──────────────────────────────────────────
print("=" * 60)
print("   HOUSE PRICE PREDICTION - ML Models Comparison")
print("=" * 60)

# Using California Housing dataset (real-world data, no download needed)
housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.rename(columns={'MedHouseVal': 'Price'}, inplace=True)

print("\n📦 Dataset Loaded Successfully!")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 3. EXPLORATORY DATA ANALYSIS (EDA) ───────────────────────
print("\n📊 Basic Info:")
print(df.describe().round(2))

print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Plot 1: Distribution of House Prices
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Price'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of House Prices')
plt.xlabel('Price (in $100,000s)')
plt.ylabel('Frequency')

# Plot 2: Correlation Heatmap
plt.subplot(1, 2, 2)
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ EDA plots saved as 'eda_plots.png'")

# ── 4. FEATURE ENGINEERING ───────────────────────────────────
# Create a new feature: Rooms per Household
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerRoom']   = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

print("\n🛠️  New Features Created:")
print("   → RoomsPerHousehold, BedroomsPerRoom, PopulationPerHousehold")

# ── 5. PREPARE DATA ───────────────────────────────────────────
X = df.drop('Price', axis=1)
y = df['Price']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✂️  Data Split:")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing samples  : {X_test.shape[0]}")

# ── 6. DEFINE & TRAIN MODELS ──────────────────────────────────
models = {
    "Linear Regression"         : LinearRegression(),
    "Ridge Regression"          : Ridge(alpha=1.0),
    "Lasso Regression"          : Lasso(alpha=0.01),
    "Decision Tree"             : DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest"             : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting"         : GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor"  : SVR(kernel='rbf', C=100, epsilon=0.1),
}

print("\n🤖 Training Models...\n")
results = []

for name, model in models.items():
    # SVR works better with scaled data
    if name == "Support Vector Regressor":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results.append({
        'Model'  : name,
        'MAE'    : round(mae, 4),
        'RMSE'   : round(rmse, 4),
        'R² Score': round(r2, 4)
    })
    print(f"   ✔ {name:<30} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

# ── 7. RESULTS COMPARISON ─────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)

print("\n" + "=" * 60)
print("   MODEL PERFORMANCE COMPARISON (sorted by R² Score)")
print("=" * 60)
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]['Model']
best_r2    = results_df.iloc[0]['R² Score']
print(f"\n🏆 Best Model: {best_model}  (R² = {best_r2})")

# ── 8. VISUALIZATION: Model Comparison ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#00BCD4', '#795548']

for ax, metric in zip(axes, ['R² Score', 'MAE', 'RMSE']):
    bars = ax.barh(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(f'Model Comparison — {metric}', fontsize=13, fontweight='bold')
    ax.set_xlabel(metric)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Model comparison chart saved as 'model_comparison.png'")

# ── 9. FEATURE IMPORTANCE (Random Forest) ─────────────────────
rf_model = models["Random Forest"]
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 5))
feat_imp.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance chart saved as 'feature_importance.png'")

# ── 10. PREDICT ON NEW DATA (Demo) ────────────────────────────
print("\n" + "=" * 60)
print("   SAMPLE PREDICTION USING BEST MODEL (Random Forest)")
print("=" * 60)

# Sample input (you can change these values)
sample = pd.DataFrame([{
    'MedInc'               : 5.0,
    'HouseAge'             : 20.0,
    'AveRooms'             : 6.0,
    'AveBedrms'            : 1.2,
    'Population'           : 1200.0,
    'AveOccup'             : 3.0,
    'Latitude'             : 34.0,
    'Longitude'            : -118.0,
    'RoomsPerHousehold'    : 2.0,
    'BedroomsPerRoom'      : 0.2,
    'PopulationPerHousehold': 400.0
}])

predicted_price = models["Random Forest"].predict(sample)[0]
print(f"\n   Input Features    : Median Income=5.0, HouseAge=20, Rooms=6 ...")
print(f"   Predicted Price   : ${predicted_price * 100_000:,.0f}")

