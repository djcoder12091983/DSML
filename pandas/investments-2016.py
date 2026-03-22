import pandas as pd

def find_investments(insurance: pd.DataFrame) -> pd.DataFrame:
    c1 = insurance.groupby('tiv_2015')['pid'].count()
    c1 = c1[c1 > 1].reset_index()['tiv_2015']
    #print(c1)
    c2 = insurance.groupby(['lat', 'lon'])['pid'].count()
    c2 = c2[c2 > 1].reset_index()
    c2 = c2.apply(lambda row: str(row.lat) + '#' + str(row.lon), axis = 1)
    #print(c2)
    insurance['location'] = insurance.apply(lambda row: str(row.lat) + '#' + str(row.lon), axis = 1)
    #print(insurance)
    res = insurance[insurance['tiv_2015'].isin(c1) & ~insurance['location'].isin(c2)]
    #print(res)
    return pd.DataFrame({'tiv_2016': [round(res['tiv_2016'].sum(),2)]})