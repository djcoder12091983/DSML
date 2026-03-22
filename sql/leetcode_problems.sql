# second-highest-salary
select (select distinct salary from Employee order by salary desc LIMIT 1 OFFSET 1) as SecondHighestSalary

# rank-scores
select score, DENSE_RANK() OVER(order by score DESC) as 'rank' from Scores

# consecutive-numbers
with t1 as (select id,
    lag(id, 1) OVER (partition by num order by id) as prev_1,
    lag(id, 2) OVER (partition by num order by id) as prev_2,
    num
from Logs)
select distinct num as ConsecutiveNums from t1 where prev_2 + 1 = prev_1 and prev_1 + 1 = id

# department-highest-salary
select t2.name as Department, t1.name as Employee, t3.max_sal as Salary from
Employee t1 join Department t2 join (select departmentId, max(salary) as max_sal from Employee group by departmentId) t3
on t1.departmentId = t2.id and t1.salary = t3.max_sal and t2.id = t3.departmentId

# managers-with-at-least-5-direct-reports
select mname as name from
(select e.id as empid, m.id as mid, m.name as mname from Employee e join Employee m where e.managerId = m.id) t
group by mid having count(empid) > 4

# investments-in-2016
select round(sum(tiv_2016),2) as tiv_2016 from Insurance
where 
tiv_2015 in (select tiv_2015 from Insurance group by tiv_2015 having count(pid) > 1)
and concat(lat, lon, '#') not in
(select concat(lat, lon, '#') from Insurance group by lat, lon having count(pid) > 1)

# friend-requests-ii-who-has-the-most-friends
with t1 as (select requester_id as id, count(accepter_id) as num from RequestAccepted group by requester_id)
,t2 as (select accepter_id as id, count(requester_id) as num from RequestAccepted group by accepter_id)

select id, num from
(select t1.id as id, t1.num+IFNULL(t2.num,0) as num from t1 left join t2 on t1.id = t2.id
union
select t2.id as id, IFNULL(t1.num,0)+t2.num as num from t1 right join t2 on t1.id = t2.id) t
order by num desc limit 1 offset 0

# tree-node
select id, 

# determine type
case when p_id is NULL then 'Root' when ifnull(children, 0) > 0 then 'Inner' else 'Leaf' end as type 

from Tree t1 left join
(select p_id as node, count(id) as children from Tree group by p_id) t2 on t1.id = t2.node

# customers-who-bought-all-products
select customer_id from
(select distinct customer_id, product_key from Customer) t
group by customer_id having count(product_key) = (select distinct count(product_key) as count from Product)

# product-sales-analysis-iii
select t1.product_id, first_year, quantity, price
from Sales t1 join
(select product_id, min(year) as first_year from Sales group by product_id) t2
on t1.product_id = t2.product_id and t1.year = t2.first_year

# market-analysis-i
select user_id as buyer_id, join_date, ifnull(buy_count, 0) as orders_in_2019
from Users t1 left join 
(select buyer_id, count(item_id) as buy_count from Orders where YEAR(order_date) = 2019 group by buyer_id) t2
on t1.user_id = t2.buyer_id

# product-price-at-a-given-date
select t1.product_id, ifnull(t2.new_price, 10) as price
from (select distinct product_id from Products) t1 left join
(select p.product_id, p.new_price
from Products p join
(select product_id, max(change_date) as date from Products
where change_date <= '2019-08-16'
group by product_id) t
on p.product_id = t.product_id and p.change_date = t.date) t2
on t1.product_id = t2.product_id

# count-salary-categories
with category_count as
(select 
    case 
    when income < 20000 then 'Low Salary' when income >= 20000 and income <= 50000 then 'Average Salary' else 'High Salary' end
    as category from Accounts)

select 'Low Salary' as category, count(*) as accounts_count from category_count where category = 'Low Salary' 
union all
select 'Average Salary' as category, count(*) as accounts_count from category_count where category = 'Average Salary'
union all
select 'High Salary' as category, count(*) as accounts_count from category_count where category = 'High Salary'