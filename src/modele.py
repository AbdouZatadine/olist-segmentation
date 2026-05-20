import joblib
from sklearn.cluster import KMeans

def entrainer_kmeans(df_rfm_new, k=5):
    """Entrainer le modèle K-Means"""

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_rfm_new)

    return kmeans


def ajouter_segments(df_rfm, kmeans):
    """Ajouter les segments au tableau RFM"""

    noms_segments = {
        0: 'Inactifs',
        1: 'Nouveaux',
        2: 'Mécontents',
        3: 'Fidèles',
        4: 'Gros acheteurs'
    }

    df_rfm['segment'] = kmeans.labels_
    df_rfm['segment_nom'] = df_rfm['segment'].map(noms_segments)

    return df_rfm


def sauvegarder_modele(kmeans, chemin):
    """Sauvegarder le modèle K-Means"""

    joblib.dump(kmeans, chemin)
    print("Modele sauvegardé avec succes !")