import pandas as pd

def most_friends(request_accepted: pd.DataFrame) -> pd.DataFrame:
    t1 = request_accepted.groupby(by='requester_id')['accepter_id'].count()
    t2 = request_accepted.groupby(by='accepter_id')['requester_id'].count()
    index = pd.concat([request_accepted['accepter_id'], request_accepted['requester_id']]).unique()
    res = pd.DataFrame({'count1': t1, 'count2': t2}, index = index).fillna(0)
    res['num'] = res['count1'] + res['count2']
    max_data = res.sort_values(by = 'num', ascending = False).head(1)
    return pd.DataFrame({'id': max_data.index, 'num': max_data['num']})
    