import pandas as pd

def charger_donnees(chemin):
    """Charger les 4 fichiers CSV"""
    df_customers = pd.read_csv(chemin + 'olist_customers_dataset.csv')
    df_orders = pd.read_csv(chemin + 'olist_orders_dataset.csv')
    df_payments = pd.read_csv(chemin + 'olist_order_payments_dataset.csv')
    df_reviews = pd.read_csv(chemin + 'olist_order_reviews_dataset.csv')
    return df_customers, df_orders, df_payments, df_reviews


def nettoyer_donnees(df_customers, df_orders, df_payments, df_reviews):
    """nettoyer et fusionner le données"""

    df_customers = df_customers[['customer_id', 'customer_unique_id']]
    df_orders = df_orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp']]
    df_payments = df_payments[['order_id', 'payment_value']]
    df_reviews = df_reviews[['order_id', 'review_score']]


    df_payments = df_payments.drop_duplicates()
    df_reviews = df_reviews.drop_duplicates()


    df_orders = df_orders[df_orders['order_status'] == 'delivered']

  
    df_orders['order_purchase_timestamp'] = pd.to_datetime(df_orders['order_purchase_timestamp'])


    df_payments_grouped = df_payments.groupby('order_id')['payment_value'].sum().reset_index()
    df_reviews_grouped = df_reviews.groupby('order_id')['review_score'].mean().reset_index()


    df = df_orders.merge(df_customers, on='customer_id', how='left')
    df = df.merge(df_payments_grouped, on='order_id', how='left')
    df = df.merge(df_reviews_grouped, on='order_id', how='left')


    df = df.drop(columns=['order_status'])

    df = df.dropna(subset=['payment_value'])
    df['review_score'] = df['review_score'].fillna(df['review_score'].mean())

    return df