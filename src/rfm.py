import pandas as pd
from sklearn.preprocessing import StandardScaler

def calculer_rfm(df):
    """Calculer le RFM à partir du dataset fusionné"""

    # Date de reference
    date_reference = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    # Calcul RFM
    df_rfm = df.groupby('customer_unique_id').agg(
        recency      = ('order_purchase_timestamp', lambda x: (date_reference - x.max()).days),
        frequency    = ('order_id', 'count'),
        monetary     = ('payment_value', 'sum'),
        satisfaction = ('review_score', 'mean')
    ).reset_index()

    return df_rfm


def normaliser(df_rfm):
    """Normaliser les données RFM pour le modèle"""

    scaler = StandardScaler()
    df_rfm_new = df_rfm[['recency', 'frequency', 'monetary', 'satisfaction']].copy()
    df_rfm_new = pd.DataFrame(
        scaler.fit_transform(df_rfm_new),
        columns=['recency', 'frequency', 'monetary', 'satisfaction']
    )

    return df_rfm_new, scaler