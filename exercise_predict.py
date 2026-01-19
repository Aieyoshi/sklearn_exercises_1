import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("datasets/train_data_us.csv")

df.loc[df["last_price"] > 113000, "price_class"] = 1
df.loc[df["last_price"] <= 113000, "price_class"] = 0

features = df.drop(["last_price", "price_class"], axis=1)
target = df["price_class"]

model = DecisionTreeClassifier()
model.fit(features, target)

new_features = pd.DataFrame(
    [
        [None, None, 2.8, 25, None, 25, 0, 0, 0, None, 0, 30706.0, 7877.0],
        [None, None, 2.75, 25, None, 25, 0, 0, 0, None, 0, 36421.0, 9176.0],
    ],
    columns=features.columns,
)

new_features.loc[0, "bedrooms"] = 12
new_features.loc[0, "total_area"] = 900.0
new_features.loc[0, "living_area"] = 409.7
new_features.loc[0, "kitchen_area"] = 112.0

new_features.loc[1, "bedrooms"] = 2
new_features.loc[1, "total_area"] = 109.0
new_features.loc[1, "living_area"] = 32.0
new_features.loc[1, "kitchen_area"] = 40.5

predictions = model.predict(new_features)

labels = ["barato (0)", "caro (1)"]
for i, p in enumerate(predictions.astype(int)):
    print(f"Apartamento {i+1}: {labels[p]}")
