# Executed SQL queries

## 01_in_stock_where_limit

```sql
SELECT title, price_gbp, rating FROM books
        WHERE in_stock = 1 ORDER BY price_gbp DESC LIMIT 10
```

| title                                                                  |   price_gbp |   rating |
|:-----------------------------------------------------------------------|------------:|---------:|
| Boar Island (Anna Pigeon #19)                                          |       59.48 |        3 |
| Candide                                                                |       58.63 |        3 |
| The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) |       57.7  |        4 |
| Animal Farm                                                            |       57.22 |        3 |
| The Past Never Ends                                                    |       56.5  |        4 |
| The Last Painting of Sara de Vos                                       |       55.55 |        2 |
| A Flight of Arrows (The Pathfinders #2)                                |       55.53 |        5 |
| Alice in Wonderland (Alice's Adventures in Wonderland #1)              |       55.53 |        1 |
| Murder at the 42nd Street Library (Raymond Ambler #1)                  |       54.36 |        4 |
| The Last Mile (Amos Decker #2)                                         |       54.21 |        2 |

## 02_distinct_categories

```sql
SELECT DISTINCT category_name FROM categories ORDER BY category_name
```

| category_name      |
|:-------------------|
| Classics           |
| Historical Fiction |
| Mystery            |

## 03_rating_between

```sql
SELECT title, rating, price_inr FROM books
        WHERE rating BETWEEN 4 AND 5 ORDER BY rating DESC, title LIMIT 10
```

| title                                                                    |   rating |   price_inr |
|:-------------------------------------------------------------------------|---------:|------------:|
| A Flight of Arrows (The Pathfinders #2)                                  |        5 |     5858.42 |
| A Spy's Devotion (The Regency Spies of London #1)                        |        5 |     1790.33 |
| A Time of Torment (Charlie Parker #14)                                   |        5 |     5100.92 |
| Between Shades of Gray                                                   |        5 |     2193.34 |
| Mrs. Houdini                                                             |        5 |     3191.38 |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |     5517.65 |
| The Girl You Lost                                                        |        5 |     1296.59 |
| The Passion of Dolssa                                                    |        5 |     2987.76 |
| The Red Tent                                                             |        5 |     3762.13 |
| The Silkworm (Cormoran Strike #2)                                        |        5 |     2431.78 |

## 04_category_in

```sql
SELECT c.category_name, COUNT(*) AS book_count
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        WHERE c.category_name IN ('Mystery', 'Classics')
        GROUP BY c.category_name ORDER BY c.category_name
```

| category_name   |   book_count |
|:----------------|-------------:|
| Classics        |           19 |
| Mystery         |           32 |

## 05_join_top_rated

```sql
SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, c.category_name, b.title LIMIT 10
```

| category_name      | title                                                                    |   rating |   price_gbp |
|:-------------------|:-------------------------------------------------------------------------|---------:|------------:|
| Historical Fiction | A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 |
| Historical Fiction | A Spy's Devotion (The Regency Spies of London #1)                        |        5 |       16.97 |
| Historical Fiction | Between Shades of Gray                                                   |        5 |       20.79 |
| Historical Fiction | Mrs. Houdini                                                             |        5 |       30.25 |
| Historical Fiction | The Passion of Dolssa                                                    |        5 |       28.32 |
| Historical Fiction | The Red Tent                                                             |        5 |       35.66 |
| Historical Fiction | Voyager (Outlander #3)                                                   |        5 |       21.07 |
| Historical Fiction | While You Were Mine                                                      |        5 |       41.32 |
| Mystery            | A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 |
| Mystery            | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  |

## 06_category_average

```sql
SELECT c.category_name, COUNT(*) AS book_count,
               ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
        FROM books AS b JOIN categories AS c ON b.category_id = c.category_id
        GROUP BY c.category_name ORDER BY average_price_gbp DESC
```

| category_name      |   book_count |   average_price_gbp |
|:-------------------|-------------:|--------------------:|
| Classics           |           19 |               36.55 |
| Historical Fiction |           26 |               33.64 |
| Mystery            |           32 |               31.72 |

## pandas merge of query 05

| category_name      | title                                                                    |   rating |   price_gbp |
|:-------------------|:-------------------------------------------------------------------------|---------:|------------:|
| Historical Fiction | A Flight of Arrows (The Pathfinders #2)                                  |        5 |       55.53 |
| Historical Fiction | A Spy's Devotion (The Regency Spies of London #1)                        |        5 |       16.97 |
| Historical Fiction | Between Shades of Gray                                                   |        5 |       20.79 |
| Historical Fiction | Mrs. Houdini                                                             |        5 |       30.25 |
| Historical Fiction | The Passion of Dolssa                                                    |        5 |       28.32 |
| Historical Fiction | The Red Tent                                                             |        5 |       35.66 |
| Historical Fiction | Voyager (Outlander #3)                                                   |        5 |       21.07 |
| Historical Fiction | While You Were Mine                                                      |        5 |       41.32 |
| Mystery            | A Time of Torment (Charlie Parker #14)                                   |        5 |       48.35 |
| Mystery            | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |        5 |       52.3  |

## SQL and pandas side by side

| SQL category_name   | SQL title                                                                |   SQL rating |   SQL price_gbp | pandas category_name   | pandas title                                                             |   pandas rating |   pandas price_gbp |
|:--------------------|:-------------------------------------------------------------------------|-------------:|----------------:|:-----------------------|:-------------------------------------------------------------------------|----------------:|-------------------:|
| Historical Fiction  | A Flight of Arrows (The Pathfinders #2)                                  |            5 |           55.53 | Historical Fiction     | A Flight of Arrows (The Pathfinders #2)                                  |               5 |              55.53 |
| Historical Fiction  | A Spy's Devotion (The Regency Spies of London #1)                        |            5 |           16.97 | Historical Fiction     | A Spy's Devotion (The Regency Spies of London #1)                        |               5 |              16.97 |
| Historical Fiction  | Between Shades of Gray                                                   |            5 |           20.79 | Historical Fiction     | Between Shades of Gray                                                   |               5 |              20.79 |
| Historical Fiction  | Mrs. Houdini                                                             |            5 |           30.25 | Historical Fiction     | Mrs. Houdini                                                             |               5 |              30.25 |
| Historical Fiction  | The Passion of Dolssa                                                    |            5 |           28.32 | Historical Fiction     | The Passion of Dolssa                                                    |               5 |              28.32 |
| Historical Fiction  | The Red Tent                                                             |            5 |           35.66 | Historical Fiction     | The Red Tent                                                             |               5 |              35.66 |
| Historical Fiction  | Voyager (Outlander #3)                                                   |            5 |           21.07 | Historical Fiction     | Voyager (Outlander #3)                                                   |               5 |              21.07 |
| Historical Fiction  | While You Were Mine                                                      |            5 |           41.32 | Historical Fiction     | While You Were Mine                                                      |               5 |              41.32 |
| Mystery             | A Time of Torment (Charlie Parker #14)                                   |            5 |           48.35 | Mystery                | A Time of Torment (Charlie Parker #14)                                   |               5 |              48.35 |
| Mystery             | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |            5 |           52.3  | Mystery                | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) |               5 |              52.3  |

SQL JOIN and pandas merge identical: **True**.
