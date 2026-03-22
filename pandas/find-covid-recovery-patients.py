import pandas as pd

def find_covid_recovery_patients(patients: pd.DataFrame, covid_tests: pd.DataFrame) -> pd.DataFrame:
    
    def recovery_time(group):
        #print(group)
        pos = group[group['result'] == 'Positive']
        if len(pos) == 0:
            # not possible
            return -1
        
        pos_date = pd.to_datetime(pos.iloc[0]['test_date'])
        #print("positive_date: ", pos_date)
        neg = group[(pd.to_datetime(group['test_date']) >= pos_date) & (group['result'] == 'Negative')]
        if len(neg) == 0:
            # not possible
            return -1

        #print("negative_date: ", neg.iloc[0, 2])
        #print(type(pos_date))
        diff = pd.to_datetime(neg.iloc[0]['test_date']) - pos_date
        #print("diff_days: ", diff.days)
        return diff.days
    
    recovery_data = covid_tests.sort_values(by = ['patient_id', 'test_date']).groupby('patient_id').apply(recovery_time)
    ans = patients.set_index('patient_id', drop = False)
    ans['recovery_time'] = recovery_data
    #print(patients)

    ans = ans.sort_values(by = ['recovery_time', 'patient_name'])
    return ans[ans['recovery_time'] != -1]