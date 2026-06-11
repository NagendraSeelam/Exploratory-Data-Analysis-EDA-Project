import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load Dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Statistical Summary
print("Dataset Shape:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Visualization 1
df.hist(figsize=(8,6))
plt.suptitle("Iris Dataset Feature Distribution")
plt.savefig("feature_distribution.png")
plt.show()

# Visualization 2
correlation = df.corr()

plt.imshow(correlation, cmap="Blues")
plt.colorbar()
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

print("EDA Completed Successfully")
