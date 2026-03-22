import pandas as pd

def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    max_sal = employee.groupby('departmentId')['salary'].max()
    dept_max_sal = department.set_index('id')
    dept_max_sal['salary'] = max_sal
    #print(dept_max_sal)

    data = pd.merge(employee, dept_max_sal, left_on = 'departmentId', right_on = 'id')
    data = data[data['salary_x'] == data['salary_y']]
    return data.rename(columns = {'name_x': 'Employee', 'name_y' : 'Department', 'salary_x' : 'Salary'})[['Department', 'Employee', 'Salary']]
    