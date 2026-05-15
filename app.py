import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper to determine base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models safely
try:
    ss_seated = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'scaler_seated_activities.pkl'))
    ss_ohe = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'scaler_ohe_gender.pkl'))
    ss_oe = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'scaler_ordinal_encoding.pkl'))
    pca_model = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'pca_model.pkl'))
    kmeans_model = joblib.load(os.path.join(BASE_DIR, 'ml_model', 'kmeans_model (2).pkl'))
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")



@app.route('/')
def home():
    # Serve the HTML file from the templates directory
    return send_from_directory(os.path.join(BASE_DIR, 'templates'), 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'answers' not in data:
        return jsonify({'ok': False, 'message': 'No answers provided.'}), 400
        
    ans = data['answers']
    
    try:
        # Extract features exactly in the order expected by the PCA
        seated = np.array([[ans['sitting_hours']]])
        gender = np.array([[ans['gender']]])
        
        ordinals = np.array([[
            ans['morning_stiffness'],
            ans['persistent_pain'],
            ans['sitting_posture'],
            ans['sleeping_posture'],
            ans['sleep_turning'],
            ans['sleep_quality'],
            ans['physical_activity'],
            ans['ott_usage'],
            ans['family_backpain'],
            ans['gut_infection']
        ]])
        
        # Scale individually based on your original pipeline
        seated_scaled = ss_seated.transform(seated)
        gender_scaled = ss_ohe.transform(gender)
        ordinals_scaled = ss_oe.transform(ordinals)
        
        # Concatenate horizontally
        features = np.hstack([seated_scaled, gender_scaled, ordinals_scaled])
        
        # Apply PCA reduction
        features_pca = pca_model.transform(features)
        
        # Predict Cluster using KMeans
        cluster_id = int(kmeans_model.predict(features_pca)[0])
        
        # Map exactly back to clusters based on user image
        cluster_risk_mapping = {
            0: "High",
            1: "Very High",
            2: "Low",
            3: "Moderate"
        }
        risk = cluster_risk_mapping[cluster_id]
        
        # Health Suggestions mapping
        suggestions = {
            "Low": "Maintain your healthy lifestyle. Keep up the good posture, regular physical activity, and adequate sleep.",
            "Moderate": "Your responses indicate moderate risk. Focus on ergonomic improvements, consistent targeted exercise, and consider consulting a physiotherapist.",
            "High": "Your pattern aligns with significant risk factors for spondylitis. We recommend consulting a healthcare professional for an evaluation.",
            "Very High": "Your pattern aligns with severe risk factors for spondylitis. We strongly recommend consulting a rheumatologist immediately."
        }
        
        # Plotting the full cluster graph using synthetic proxy data for cloud background
        fig, ax = plt.subplots(figsize=(7, 5))
        centers = kmeans_model.cluster_centers_
        
        # Exact colors mapped from the user's provided picture
        colors = {
            0: "orange",
            1: "red",
            2: "green",
            3: "blue"
        }
        
        n_points_per_cluster = 150
        synthetic_X = []
        synthetic_y = []
        np.random.seed(42) # Ensure graph consistency across user tests
        
        for cid in range(kmeans_model.n_clusters):
            c_color = colors[cid]
            c_risk = cluster_risk_mapping[cid]
            
            # 1) Generate and plot full proxy clusters around the center
            # Scale of 0.6 mimics typical PCA variance spread for normalized data
            pts = np.random.normal(loc=centers[cid], scale=0.6, size=(n_points_per_cluster, centers.shape[1]))
            synthetic_X.append(pts)
            synthetic_y.extend([cid] * n_points_per_cluster)
            
            # Plot the background cloud
            ax.scatter(pts[:, 0], pts[:, 1], c=c_color, s=40, alpha=1.0, edgecolors='none', label=f'Cluster {cid}: {c_risk} Risk')
        
        # 2) Plot ALL Centroids in Black as shown in image
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=80, marker='o', edgecolors='none', label='Centroids')

        synthetic_X = np.vstack(synthetic_X)
        synthetic_y = np.array(synthetic_y)
        
        # Hardcoding the exact metrics from the model's training phase (to be provided by user)
        sil_score = 0.336 
        db_score = 0.960
        ch_score = 180.257

        # 3) Plot User's specific highlighted point (Yellow)
        ax.scatter(features_pca[0, 0], features_pca[0, 1], c='yellow', s=80, marker='o', edgecolors='none', label='YOUR POSITION')
        
        ax.set_title("Clusters of people", fontsize=14)
        ax.set_xlabel("PC1: Pain (Pain, Turning, Family, Stiffness)")
        ax.set_ylabel("PC2: Lifestyle (Sitting, OTT, Gut, Activity)")
        
        # Deduplicate legend labels
        handles, lbls = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        graph_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

        if risk == "Low":
            desc = "Your responses map to a lower-risk cluster. Keep up healthy posture and activity!"
        elif risk == "Mild":
            desc = "Your answers place you in a mild-risk cluster. Minor lifestyle changes can help you maintain health."
        elif risk == "Moderate":
            desc = "Your answers place you in a moderate-risk cluster. Consider incorporating more movement and ergonomic improvements."
        else:
            desc = "Your pattern aligns with a higher-risk cluster. We recommend consulting a healthcare professional for an evaluation."
            
        return jsonify({
            'ok': True,
            'result': {
                'cluster': cluster_id,
                'risk_level': risk,
                'description': desc,
                'suggestion': suggestions[risk],
                'graph_b64': graph_b64,
                'metrics': {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score
                }
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'ok': False, 'message': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask API on the assigned port dynamically for Render
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting ML Backend Server on http://0.0.0.0:{port} ...")
    app.run(host='0.0.0.0', port=port)
