import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.sparse import hstack

# Load your data
data = pd.read_csv("combined_data_with_sentiment.csv")

# Drop rows with NaN values in 'cleaned_reviews' or impute missing sentiment scores if needed
data = data.dropna(subset=['cleaned_reviews', 'sentiment_score'])  # Ensure no missing reviews or sentiment

# Aggregate sentiment scores for each course
course_sentiment = data.groupby('course_id')['sentiment_score'].mean().reset_index()
course_sentiment.columns = ['course_id', 'avg_sentiment_score']

# Merge aggregated sentiment scores back to the original data
data = data.merge(course_sentiment, on='course_id', how='left')

# Handle any remaining missing values
data = data.fillna({'avg_sentiment_score': data['avg_sentiment_score'].mean()})  # Impute missing avg_sentiment_score with mean

# Sample 1% of the dataset randomly
sampled_data = data.sample(frac=0.01, random_state=42)

# Preprocessing text data (cleaned_reviews)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(sampled_data['cleaned_reviews'])

# Preprocessing numeric features (rating, review_count, sentiment_score, avg_rating, avg_sentiment_score)
numeric_features = sampled_data[['rating', 'review_count', 'sentiment_score', 'avg_rating', 'avg_sentiment_score']]
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Combine TF-IDF and numeric features into a single feature matrix
combined_features = hstack([tfidf_matrix, numeric_features_scaled])

# Clustering with KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(combined_features)
sampled_data['kmeans_cluster'] = kmeans.labels_

# Clustering with DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(combined_features)
sampled_data['dbscan_cluster'] = dbscan_labels

# Clustering with Spectral Clustering
similarity_matrix = cosine_similarity(combined_features)

# Ensure no NaN values in similarity matrix
if np.any(np.isnan(similarity_matrix)):
    similarity_matrix = np.nan_to_num(similarity_matrix)  # Replace NaN with 0

spectral_clustering = SpectralClustering(n_clusters=5, affinity='precomputed')
spectral_labels = spectral_clustering.fit_predict(similarity_matrix)
sampled_data['spectral_cluster'] = spectral_labels

# Function to recommend courses based on clustering
def recommend_courses(course_index, cluster_column='kmeans_cluster', num_recommendations=5):
    cluster_id = sampled_data.iloc[course_index][cluster_column]
    similar_courses = sampled_data[sampled_data[cluster_column] == cluster_id]
    return similar_courses[['name', 'institution', 'rating', 'avg_sentiment_score']].head(num_recommendations)

# Example: Recommend 5 courses similar to course at index 0 based on KMeans cluster
recommended_courses_kmeans = recommend_courses(0, cluster_column='kmeans_cluster')
print("KMeans Recommendations:")
print(recommended_courses_kmeans)

# Visualize the clusters for KMeans (using rating and review_count for simplicity)
plt.scatter(sampled_data['rating'], sampled_data['review_count'], c=sampled_data['kmeans_cluster'], cmap='viridis')
plt.xlabel('Rating')
plt.ylabel('Review Count')
plt.title('KMeans Clustering')
plt.show()

# Visualize the clusters for DBSCAN
plt.scatter(sampled_data['rating'], sampled_data['review_count'], c=sampled_data['dbscan_cluster'], cmap='viridis')
plt.xlabel('Rating')
plt.ylabel('Review Count')
plt.title('DBSCAN Clustering')
plt.show()

# Visualize the clusters for Spectral Clustering
plt.scatter(sampled_data['rating'], sampled_data['review_count'], c=sampled_data['spectral_cluster'], cmap='viridis')
plt.xlabel('Rating')
plt.ylabel('Review Count')
plt.title('Spectral Clustering')
plt.show()

# Evaluate clustering performance using silhouette score
silhouette_kmeans = silhouette_score(combined_features, kmeans.labels_)
silhouette_dbscan = silhouette_score(combined_features, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
silhouette_spectral = silhouette_score(combined_features, spectral_labels)

print(f"Silhouette Score for KMeans: {silhouette_kmeans}")
print(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")
print(f"Silhouette Score for Spectral Clustering: {silhouette_spectral}")