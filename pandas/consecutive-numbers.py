import pandas as pd

def consecutive_numbers(logs: pd.DataFrame) -> pd.DataFrame:

    def check_consecutive(group_row):
        seq = group_row['id'].sort_values().to_list()
        #print(seq)
        n = len(seq) 
        if n < 3:
            return False
        c = 1
        for i in range(n - 1):
            if seq[i] + 1 == seq[i + 1]:
                c += 1
            else:
                c = 1
            
            if c == 3:
                return True
        return False
    
    data = logs.groupby('num').apply(check_consecutive)
    return pd.DataFrame({'ConsecutiveNums': data[data == True].index})