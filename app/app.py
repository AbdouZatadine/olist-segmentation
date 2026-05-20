from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Chemin de base du projet
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Charger les données et le modèle avec chemins relatifs
df_rfm = pd.read_csv(os.path.join(BASE_DIR, 'donnees', 'traitees', 'df_rfm.csv'))
kmeans = joblib.load(os.path.join(BASE_DIR, 'sorties', 'modeles', 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'sorties', 'modeles', 'scaler.pkl'))

def anonymiser_id(customer_id):
    return str(customer_id)[:7] + '…'

@app.route('/')
def index():
    stats = {
        'total_clients': len(df_rfm),
        'nb_segments': df_rfm['segment_nom'].nunique(),
        'ca_total': round(df_rfm['monetary'].sum(), 2),
        'note_moyenne': round(df_rfm['satisfaction'].mean(), 2),
        'segments': df_rfm['segment_nom'].value_counts().to_dict()
    }
    
    segments = ['Nouveaux', 'Inactifs', 'Mécontents', 'Fidèles', 'Gros acheteurs']
    clients_recents = []
    
    emoji_map = {
        'Nouveaux': '🆕',
        'Inactifs': '💤',
        'Mécontents': '⚠️',
        'Fidèles': '✅',
        'Gros acheteurs': '👑'
    }
    
    for segment in segments:
        client = df_rfm[df_rfm['segment_nom'] == segment].nsmallest(1, 'recency').iloc[0]
        clients_recents.append({
            'id': anonymiser_id(client['customer_unique_id']),
            'recency': f"{client['recency']} j",
            'frequency': int(client['frequency']),
            'monetary': f"R$ {client['monetary']:,.0f}",
            'segment_emoji': emoji_map[segment],
            'segment_nom': segment
        })
    
    return render_template('dashboard.html', stats=stats, clients_recents=clients_recents)

@app.route('/segments')
def segments():
    moyennes = df_rfm.groupby('segment_nom')[['recency', 'frequency', 'monetary', 'satisfaction']].mean().round(2)
    moyennes = moyennes.reset_index()
    counts = df_rfm['segment_nom'].value_counts()
    total = len(df_rfm)
    moyennes['count'] = moyennes['segment_nom'].map(counts)
    moyennes['pct'] = (moyennes['count'] / total * 100).round(1)
    moyennes = moyennes.to_dict(orient='records')

    strategie = {
        'Inactifs': 'Email de relance + Offre promotionnelle limitée',
        'Nouveaux': '-10% sur la prochaine commande + Produits similaires',
        'Mécontents': 'Comprendre le problème + Bon cadeau',
        'Fidèles': 'Programme de fidélité + Ventes privées',
        'Gros acheteurs': 'Programme VIP + Livraison prioritaire'
    }

    return render_template('segments.html', moyennes=moyennes, strategie=strategie)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array([[
        float(data['recency']),
        float(data['frequency']),
        float(data['monetary']),
        float(data['satisfaction'])
    ]])
    X_scaled = scaler.transform(X)
    segment = kmeans.predict(X_scaled)[0]

    noms = {
        0: 'Inactifs',
        1: 'Nouveaux',
        2: 'Mécontents',
        3: 'Fidèles',
        4: 'Gros acheteurs'
    }

    strategie = {
        'Inactifs': 'Email de relance + Offre promotionnelle limitée',
        'Nouveaux': '-10% sur la prochaine commande + Produits similaires',
        'Mécontents': 'Comprendre le problème + Bon cadeau',
        'Fidèles': 'Programme de fidélité + Ventes privées',
        'Gros acheteurs': 'Programme VIP + Livraison prioritaire'
    }

    segment_nom = noms[int(segment)]
    return jsonify({'segment': segment_nom, 'strategie': strategie[segment_nom]})

@app.route('/donnees')
def donnees():
    chemin = os.path.join(BASE_DIR, 'donnees', 'brutes')
    
    icons = {
        'olist_orders_dataset': ('fa-box', '#ffb627', 'rgba(255,182,39,0.12)'),
        'olist_customers_dataset': ('fa-users', '#56CCF2', 'rgba(86,204,242,0.12)'),
        'olist_order_items_dataset': ('fa-cart-shopping', '#6c74f5', 'rgba(108,116,245,0.12)'),
        'olist_order_payments_dataset': ('fa-credit-card', '#1de9b6', 'rgba(29,233,182,0.12)'),
        'olist_order_reviews_dataset': ('fa-star', '#ffb627', 'rgba(255,182,39,0.12)'),
        'olist_products_dataset': ('fa-tag', '#ff5c5c', 'rgba(255,92,92,0.12)'),
        'olist_sellers_dataset': ('fa-store', '#6c74f5', 'rgba(108,116,245,0.12)'),
        'olist_geolocation_dataset': ('fa-map-location-dot', '#1de9b6', 'rgba(29,233,182,0.12)'),
        'product_category_name_translation': ('fa-language', '#56CCF2', 'rgba(86,204,242,0.12)'),
    }

    fichiers = []
    for f in os.listdir(chemin):
        if f.endswith('.csv'):
            nom = f.replace('.csv', '')
            df_temp = pd.read_csv(os.path.join(chemin, f))
            icon, couleur, bg = icons.get(nom, ('fa-file-csv', '#56CCF2', 'rgba(86,204,242,0.12)'))
            fichiers.append({
                'nom': nom,
                'lignes': len(df_temp),
                'colonnes': len(df_temp.columns),
                'icon': icon,
                'couleur': couleur,
                'bg': bg
            })

    return render_template('donnees.html', fichiers=fichiers)

@app.route('/parametres')
def parametres():
    return render_template('parametres.html')

if __name__ == '__main__':
    app.run(debug=True)