import pandas as pd

MAX_DATE = pd.to_datetime('2019-08-16')

def get_price(group):
    max_date = group['change_date'].max()
    return group[group['change_date'] == max_date]['new_price'].iat[0]

def price_at_given_date(products: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame({'product_id': products['product_id'].unique()})
    new_data = products[products['change_date'] <= MAX_DATE]
    price_data = new_data.groupby('product_id').apply(get_price)
    res_data = pd.DataFrame(data = {'price': price_data})
    return pd.merge(res, res_data, on = 'product_id', how = 'left').fillna(10)