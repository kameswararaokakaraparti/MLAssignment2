import pandas as pd
import os

DATA_FILE_PATH = os.path.join("data", "diabetes_health_indicators.csv")
# Load your dataset
df = pd.read_csv(DATA_FILE_PATH)

# Separate classes
df_0 = df[df["Diabetes_binary"] == 0]
df_1 = df[df["Diabetes_binary"] == 1]

print("Class 0 count:", len(df_0))  # 218,334
print("Class 1 count:", len(df_1))  # 35,346

# Shuffle for randomness
df_0 = df_0.sample(frac=1, random_state=42).reset_index(drop=True)
df_1 = df_1.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into two equal parts
df_0_part1 = df_0.iloc[:174667]
df_0_part2 = df_0.iloc[174667:218334]

df_1_part1 = df_1.iloc[:17673]
df_1_part2 = df_1.iloc[28276:35346]

# Create two balanced files
file1 = pd.concat([df_0_part1, df_1_part1]).sample(frac=1, random_state=42).reset_index(drop=True)
file2 = pd.concat([df_0_part2, df_1_part2]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save them
file1.to_csv("balanced_file1.csv", index=False)
file2.to_csv("balanced_file2.csv", index=False)

print("Created balanced_file1.csv and balanced_file2.csv")

