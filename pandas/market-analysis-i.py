import pandas as pd

def market_analysis(users: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    count_data = orders[orders['order_date'].dt.year == 2019].groupby('buyer_id')['item_id'].count()
    users_data = users.set_index('user_id', drop = False)
    res = users_data.join(count_data, how = 'left').fillna(value = 0)
    return res[['user_id', 'join_date', 'item_id']].rename(columns = {'user_id': 'buyer_id', 'item_id': 'orders_in_2019'})
    