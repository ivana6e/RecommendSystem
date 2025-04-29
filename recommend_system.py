import joblib


def load_knn_model():
    knn_loaded, articles_loaded = joblib.load('knn_0.joblib')
    return knn_loaded, articles_loaded


def recommend(category_list, top_k=3):
    knn_loaded, articles_loaded = load_knn_model()
    distances, indices = knn_loaded.kneighbors(category_list, n_neighbors=len(articles_loaded))
    results = []
    for query_idx, (dist, idx) in enumerate(zip(distances, indices)):
        seen_articles = set() # preprocessor: load some data
        recommended = []
        for j, i in enumerate(idx):
            article = articles_loaded[i]
            if article not in seen_articles:
                seen_articles.add(article)
                recommended.append((article, dist[j]))
            if len(recommended) == top_k:
                break
        results.append({
            "category_list": category_list[query_idx],
            "recommendations": recommended
        })
    return results


def main():
    category_list = [
        [1, 1, 7],
        [9, 9, 1],
    ]

    recommendations = recommend(category_list)

    for rec in recommendations:
        print(f"\nInput Category: {rec['category_list']}")
        for article, distance in rec['recommendations']:
            print(f" - {article} (cosine distance: {distance:.4f})")


if __name__ == "__main__":
    main()
