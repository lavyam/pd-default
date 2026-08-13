import pandas as pd
from predict import predict_pd
import estimation

ModelArtifacts = estimation.ModelArtifacts
from estimation import load_artifacts, ARTIFACTS_PATH

artifacts = load_artifacts(ARTIFACTS_PATH)

df = pd.read_csv("train-2025.csv")

# sorting so growth features have proper history
df_sorted = df.sort_values(["id", "fs_year"]).reset_index(drop=True)

row_idx = 200
single_row = df_sorted.iloc[[row_idx]].copy()
group_rows = df_sorted.iloc[:row_idx + 1].copy()

pd_single = predict_pd(single_row, artifacts=artifacts).iloc[0]
pd_group  = predict_pd(group_rows, artifacts=artifacts).iloc[row_idx]

print("PD alone    :", pd_single)
print("PD in group :", pd_group)
print("Abs diff    :", abs(pd_single - pd_group))
