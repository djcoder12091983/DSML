# salary info
select Name from salary_info where salary > (select salary from salary_info where id = 8) order by salary desc;

# employee info-2
select Name from employee_info order by salary desc limit 5;

# order updates
select x.order_id, dish_name, order_qty from orders x, menu y,
(select order_id, dish_id from orders group by order_id, dish_id having count(*) > 1) z
where x.dish_id = y.dish_id and x.order_id = z.order_id and x.dish_id = z.dish_id limit 1 offset 1;

# perfect choice
select distinct customer_name from customers x, orders y where x.customer_id = y.customer_id and dish_id =
(select dish_id from orders group by dish_id order by count(dish_id) desc limit 1) order by customer_name;

# actors and movies
SELECT movie_title FROM movies LEFT JOIN movies_cast ON movies.movie_id = movies_cast.movie_id
WHERE movies_cast.actor_id IN (SELECT actor_id FROM movies_cast GROUP BY actor_id HAVING COUNT(actor_id) > 1)

# lost games
select x.game_id, game_name from (select game_id, game_name, count(x.developer_id) as c from games x, developers y where x.developer_id = y.developer_id
and developer_type = 'open-source' group by game_id having c < 2) x,
(select game_id, sum(times_played) as c from players group by game_id having c < 5000) y
where x.game_id = y.game_id;

# SQL U3
select name as Name from q3_employees where panid in (select panid from q3_salaries where salary = (select max(salary) from q3_salaries));

# SQL U2
select Name from q2_employees where month = 'Feb' group by age having count(name) > 1;

# short films
select movie_title , movie_year, concat(director_first_name, director_last_name) as director_name, concat(actor_first_name, actor_last_name) as actor_name, role
from movies T1
join movies_cast T2
on T1.movie_id = T2.movie_id
join movies_directors T3
on T1.movie_id = T3.movie_id
join directors T4
on T3.director_id = T4.director_id
join actors T5
on T2.actor_id = T5.actor_id
group BY T1.movie_time
having T1.movie_time = min(T1.movie_time);

# moviee character
select concat(director_first_name, director_last_name) as director_name, movie_title from directors T1,
(select director_id, movie_title from movies_directors X, movies Y where Y.movie_id = (select movie_id from movies_cast where role = 'SeanMaguire')
and X.movie_id = Y.movie_id) T2 where T1.director_id = T2.director_id;