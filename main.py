from src.preprocessing import charger_donnees, nettoyer_donnees
from src.rfm import calculer_rfm, normaliser
from src.modele import entrainer_kmeans, ajouter_segments, sauvegarder_modele

# Chemins
chemin_brutes = 'donnees/brutes/'
chemin_modele = 'sorties/modeles/kmeans_model.pkl'
chemin_rfm = 'donnees/traitees/df_rfm.csv'

# Etape 1 — Charger les données
print("Chargement des données...")
df_customers, df_orders, df_payments, df_reviews = charger_donnees(chemin_brutes)

# Etape 2 — Nettoyer et fusionner
print("Nettoyage et fusion...")
df = nettoyer_donnees(df_customers, df_orders, df_payments, df_reviews)

# Etape 3 — Calculer RFM
print("Calcul RFM...")
df_rfm = calculer_rfm(df)

# Etape 4 — Normaliser
print("Normalisation...")
df_rfm_new, scaler = normaliser(df_rfm)

# Etape 5 — Entrainer le modèle
print("Entrainement K-Means...")
kmeans = entrainer_kmeans(df_rfm_new, k=5)

# Etape 6 — Ajouter les segments
df_rfm = ajouter_segments(df_rfm, kmeans)

# Etape 7 — Sauvegarder
sauvegarder_modele(kmeans, chemin_modele)
df_rfm.to_csv(chemin_rfm, index=False)

print("Termine !")
print(df_rfm['segment_nom'].value_counts())
