import pandas as pd

def sales_analysis(sales: pd.DataFrame) -> pd.DataFrame:
    first_year_data = sales.groupby('product_id')['year'].min()
    res_data = pd.merge(sales, first_year_data, on = ['product_id', 'year']).rename(columns = {'year': 'first_year'})
    return res_data[['product_id', 'first_year', 'quantity', 'price']]