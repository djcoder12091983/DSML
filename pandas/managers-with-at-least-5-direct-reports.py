import pandas as pd

def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    count_data = employee.groupby('managerId')['id'].count()
    indexed_emp = employee.set_index('id')
    res = indexed_emp.join(count_data[count_data > 4], how = 'inner')
    return pd.DataFrame(res['name'])