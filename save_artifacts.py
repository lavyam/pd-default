import pandas as pd
import estimation

df_raw = pd.read_csv("train-2025.csv")
artifacts = estimation.train_full_pipeline(df_raw)
estimation.save_artifacts(artifacts)
