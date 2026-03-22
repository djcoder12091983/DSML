import pandas as pd

def exchange_seats(seat: pd.DataFrame) -> pd.DataFrame:
    data = seat.sort_values(by = 'id')
    n = len(data)
    for i in range(n):
        if i % 2 == 1:
            data.iloc[i - 1, 1], data.iloc[i, 1] = data.iloc[i, 1], data.iloc[i - 1, 1]
    return data