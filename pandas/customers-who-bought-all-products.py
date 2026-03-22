import pandas as pd

def find_customers(customer: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    c = product.shape[0]
    #data = customer.drop_duplicates() # based on customer_id and product_key
    count_data = customer.groupby('customer_id')['product_key'].nunique()
    res = count_data.apply(lambda x: x == c)
    return pd.DataFrame({'customer_id': count_data[res].index})