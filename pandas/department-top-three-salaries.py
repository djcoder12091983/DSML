import pandas as pd

def top_three_salaries(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    def max3_sal(data):
        return data.drop_duplicates().sort_values(ascending = False).head(3).to_list()
    
    max3_dept_sal = employee.groupby('departmentId')['salary'].apply(max3_sal)
    #print(max3_dept_sal)
    dept_sal = department.set_index('id')
    dept_sal['max3_salary'] = max3_dept_sal
    #print(dept_sal)

    res = pd.merge(employee, dept_sal, left_on = 'departmentId', right_on = 'id')
    #print(res)
    #res = res[res['salary'].isin(res['max3_salary'])][['name_x', 'name_y', 'salary']]
    res = res[res.apply(lambda row: row['salary'] in row['max3_salary'], axis = 1)][['name_y', 'name_x', 'salary']]
    #print(res)
    return res.rename(columns = {'name_y': 'Department', 'name_x' : 'Employee', 'salary': 'Salary'})
    