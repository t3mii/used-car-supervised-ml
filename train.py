from data import *
numeric_features = [
    "age",
    "odometer"
]

categorical_features = [
    "manufacturer",
    "model",
    "condition",
    "cylinders",
    "fuel",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state"
]

features = numeric_features + categorical_features

X = df[features]
y = df["price"]


#split training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


#categorical variables to numerical values
numeric_transformer =SimpleImputer(
    strategy="median"
)

preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(
         handle_unknown="infrequent_if_exist",
         min_frequency=20
     ),
     categorical_features),

    ("numerical",
     "passthrough",
     numeric_features)
])
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


#training linear reg model 
linear_model = LinearRegression()
linear_model.fit(X_train_encoded, y_train)
linear_predictions = linear_model.predict(X_test_encoded)
linear_mse = mean_squared_error(y_test, linear_predictions)

# baseline
mean_price=y_train.mean()
baseline_predictions=[mean_price] * len(y_test)
baseline_mse=mean_squared_error(
    y_test, baseline_predictions)

print(f"Naive Baseline MSE: {baseline_mse:,.2f}")
print(f"Linear Regression MSE: {linear_mse:,.2f}")
linear_improvement = (
    (baseline_mse - linear_mse) / baseline_mse)*100

print(f"MSE reduction: {linear_improvement:.2f}%") # seeing if the improvement is accurate
random_forest = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
random_forest.fit(X_train_encoded, y_train)
forest_predictions = random_forest.predict(
    X_test_encoded)
forest_mse = mean_squared_error(
    y_test, forest_predictions)
print(f"Random Forest MSE: {forest_mse:,.2f}")
