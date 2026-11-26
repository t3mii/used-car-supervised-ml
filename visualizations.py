from train import *
results = pd.DataFrame({
    "Model": ["Naive Baseline", "Linear Regression", "Random Forest"],
    "MSE": [baseline_mse, linear_mse, forest_mse]
})

print(results)

# Figure 1: Price distribution
plt.figure(figsize=(9, 5))
plt.hist(df["price"], bins=50, edgecolor="black")
plt.xlabel("Price ($)")
plt.ylabel("Number of Vehicles")
plt.title("Distribution of Used Car Prices")
plt.tight_layout()
plt.savefig("price_distribution.png", dpi=300)
plt.show()


# Figure 2: Actual vs predicted
plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    forest_predictions,
    alpha=0.15,
    s=10
)

plt.plot(
    [0, 100000],
    [0, 100000],
    color="red",
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Listing Price ($)")
plt.ylabel("Predicted Listing Price ($)")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()


# Figure 3: Model comparison
results = pd.DataFrame({
    "Model": [
        "Naive Baseline",
        "Linear Regression",
        "Random Forest"
    ],
    "MSE": [
        baseline_mse,
        linear_mse,
        forest_mse
    ]
})

plt.figure(figsize=(9, 5))

bars = plt.bar(
    results["Model"],
    results["MSE"] / 1000000,
    color=["gray", "steelblue", "seagreen"]
)

plt.bar_label(bars, fmt="%.2f", padding=3)

plt.title("Comparison of Used Car Pricing Models")
plt.ylabel("Mean Squared Error (Millions of USD²)")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()


# Figure 4: 5 Worst predictions
plot_data = worst_five.copy()

plot_data["vehicle"] = (
    plot_data["manufacturer"] + " " +
    plot_data["model"]
)

plot_data.set_index("vehicle")[
    ["actual_price", "predicted_price"]
].plot(
    kind="barh",
    figsize=(10, 6),
    color=["steelblue", "seagreen"]
)

plt.xlabel("Price ($)")
plt.ylabel("Vehicle")
plt.title("Five Largest Random Forest Prediction Errors")
plt.tight_layout()
plt.savefig("worst_five_predictions.png", dpi=300)
plt.show()