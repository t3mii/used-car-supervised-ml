from train import *
# improvement vs baseline
rf_improvement_vs_baseline = (
    (baseline_mse - forest_mse) / baseline_mse
) * 100

rf_improvement_vs_linear = (
    (linear_mse - forest_mse) / linear_mse
) * 100

print(f"RF reduction vs baseline: {rf_improvement_vs_baseline:.2f}%")
print(f"RF reduction vs linear regression: {rf_improvement_vs_linear:.2f}%")



# test results
test_results = X_test.copy()

test_results["actual_price"] = y_test.values
test_results["predicted_price"] = forest_predictions

test_results["error"] = (
    test_results["predicted_price"] - test_results["actual_price"]
)

test_results["squared_error"] = test_results["error"] ** 2
test_results["absolute_error"] = test_results["error"].abs()

print(test_results.head())
#5 biggest misses
worst_five = test_results.sort_values(
    by="squared_error",
    ascending=False
).head(5)

print(worst_five[[
    "manufacturer", "model", "age", "odometer",
    "condition", "title_status", "transmission",
    "drive", "type", "state",
    "actual_price", "predicted_price",
    "error", "squared_error"
]].to_string(index=False))



#analysis
for idx, row in worst_five.iterrows():
    matches = X_train[
        (X_train["manufacturer"] == row["manufacturer"]) &
        (X_train["model"] == row["model"])
    ]
    
    print("\n-----------------------------")
    print("Vehicle:", row["manufacturer"], row["model"])
    print("Actual price:", row["actual_price"])
    print("Predicted price:", row["predicted_price"])
    print("Absolute error:", row["absolute_error"])
    print("Training examples with same manufacturer/model:", len(matches))

    if len(matches) > 0:
        matching_prices = y_train.loc[matches.index]
        print("Median comparable training price:", matching_prices.median())
        print("Min comparable training price:", matching_prices.min())
        print("Max comparable training price:", matching_prices.max())

#total squared prediction error
total_squared_error = test_results["squared_error"].sum()

worst_five_squared_error = worst_five["squared_error"].sum()

contribution = (
    worst_five_squared_error / total_squared_error
) * 100

print(f"Contribution of five worst predictions: {contribution:.2f}%")