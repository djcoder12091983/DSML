import pandas as pd

def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    sorted_score = scores['score'].sort_values(ascending=False)
    #print(sorted_score)
    sorted_rank = sorted_score.rank(method='dense', ascending=False).convert_dtypes()
    #print(sorted_rank)
    return pd.DataFrame({'score': sorted_score, 'rank': sorted_rank})