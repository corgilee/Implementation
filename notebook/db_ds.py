'''
Implement a peer finder algorithm for customer similarity matching.
Design the similarity measure as you see fit, and justify your choice after the implementation.
You have a high-cardinality table customers with fields:
* customer_id (STRING): Unique customer identifier
* industry (STRING): Customer’s industry (Tech, Retail, Finance, etc.)
* pricing_tier (STRING): Pricing tier (A, B, C)
* avg_monthly_spend (DECIMAL): Average monthly spending
* tenure (INT): Customer tenure in months
Given a low-cardinality table query_customers with the same fields, write python code to return the top 5 peers from the customers table that are most similar to each row of query_customers.

'''

import pandas as pd

customers_rows = [
    ('cust_1',  'Tech',    'A', 12000.0, 24),
    ('cust_2',  'Retail',  'B',  5000.0, 18),
    ('cust_3',  'Tech',    'A', 15000.0, 36),
    ('cust_4',  'Finance', 'C',  8000.0, 12),
    ('cust_5',  'Retail',  'B',  4500.0, 20),
    ('cust_6',  'Tech',    'A', 11000.0, 30),
    ('cust_7',  'Finance', 'C',  7500.0, 10),
    ('cust_8',  'Tech',    'A', 13000.0, 28),
    ('cust_9',  'Retail',  'B',  6000.0, 15),
    ('cust_10', 'Finance', 'C',  9000.0, 22),
    ('cust_11', 'Tech',    'A', 12500.0, 25),
    ('cust_12', 'Retail',  'B',  5500.0, 19),
]

customers_df = (
    pd.DataFrame(
        customers_rows,
        columns=[
            'customer_id',
            'industry',
            'pricing_tier',
            'avg_monthly_spend',
            'tenure'
        ]
    )
    .set_index('customer_id')
)


### query customer tables

query_customers_df = (
    pd.DataFrame(
        [
            ('cust_12', 'Retail', 'B', 5500.0, 19),
        ],
        columns=[
            'customer_id',
            'industry',
            'pricing_tier',
            'avg_monthly_spend',
            'tenure'
        ]
    )
    .set_index('customer_id')
)


print(customers_df.head(10))


print(query_customers_df.head(10))
