import pandas as pd

df = pd.DataFrame({
    "a": [1, None, 3, None],
    "b": [1, 2, 3, 4],
    "c": [None, None, 1, 1],
    "d":[1,3, None, 1]
})



percent_missing = (df.isnull().sum()*100/len(df)).reset_index(name="missing_rate")\
      .query("missing_rate > 0")\
      .sort_values("missing_rate", ascending=False)\
      .reset_index(drop=True)


print(percent_missing)