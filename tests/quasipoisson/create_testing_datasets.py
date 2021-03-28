import pandas as pd

df = pd.DataFrame(
    {
        "x_00": [-1, 0, 1, 2, 3, 4],
        "x_01": [-6.5, 4, 2, 2, 1.4, 9],
        "x_02": [5, 1, 1, 1, 2.2, 0.5],
        "x_03": [0, 0, 1, 1, 1, 0],
        "y": [1, 2, 1, 8, 3, 4],
    }
)
df.to_csv("quasipoisson_data_basic.csv", index=False)
