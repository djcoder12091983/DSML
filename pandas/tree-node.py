import pandas as pd

def tree_node(tree: pd.DataFrame) -> pd.DataFrame:
    data = tree.set_index('id', drop = False)
    data['children'] = data.groupby('p_id')['id'].count()
    data = data.fillna(0)
    data['type'] = data.apply(lambda row: 'Root' if row['p_id'] == 0 else 'Inner' if row['children'] > 0 else 'Leaf', axis = 1)
    return data[['id', 'type']]