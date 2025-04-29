from sklearn.neighbors import NearestNeighbors
import joblib
import pandas as pd


def save_knn_model():
    # Load CSV
    df = pd.read_csv('data.csv')
    # df = pd.read_csv('preferences.csv')

    # Prepare feature matrix and article names
    categories = df[['category_1', 'category_2', 'category_3']].values
    articles = df['article'].tolist()

    # Fit KNN with cosine similarity
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(categories)

    # Save model
    joblib.dump((knn, articles), 'knn_0.joblib')

if __name__ == "__main__":
    save_knn_model()
