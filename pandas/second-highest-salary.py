import pandas as pd

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    #return employee.sort_values(by = 'salary', ascending = False).reset_index()
    data = employee['salary'].drop_duplicates().nlargest(2)
    #print(len(data))
    if len(data) < 2:
        val = None
    else:
        val = data.iloc[-1]
    return pd.DataFrame({'SecondHighestSalary' : [val]})