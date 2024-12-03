import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

# Load the data
combined_data = pd.read_csv("combined_data_preprocessed.csv")

# Select 10% of the data for testing
sample_size = int(len(combined_data) * 0.1)  # 10% of the dataset
sampled_data = combined_data.sample(n=sample_size, random_state=42)

# Feature Engineering: Calculate Average Rating, Review Count per Course
course_stats = sampled_data.groupby('course_id').agg(
    avg_rating=('rating', 'mean'),
    review_count=('rating', 'count')
).reset_index()

# Merge back into original dataframe
sampled_data = sampled_data.merge(course_stats, on='course_id', how='left')

# 5. Normalize Rating and Review Count
scaler = StandardScaler()
sampled_data[['normalized_rating', 'normalized_review_count']] = scaler.fit_transform(
    sampled_data[['avg_rating', 'review_count']]
)

# Set the device to GPU (CUDA)
device = 0  # Default GPU (change this if needed)

# Load the pre-trained sentiment analysis model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
classifier = pipeline(
    "text-classification", 
    model=MODEL_NAME, 
    device=device, 
    truncation=True, 
    padding=True, 
    max_length=512
)

# Map the model's labels to sentiment categories
label_mapping = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

# Function to classify sentiment for a batch of reviews
def classify_batch(batch):
    reviews = batch['reviews'].tolist()  # Convert reviews column to a list
    results = classifier(reviews)  # Perform sentiment classification
    # Extract sentiments and scores
    return [(label_mapping[result['label']], result['score']) for result in results]

# Process the dataset in batches for efficiency
def process_in_batches(df, batch_size=64):
    sentiment_results = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]  # Slice the DataFrame into batches
        sentiments = classify_batch(batch)  # Classify the batch
        sentiment_results.extend(sentiments)  # Append the results

        # Progress tracking
        print(f"Processed {min(start + batch_size, len(df))}/{len(df)} rows")

    return sentiment_results

# Apply sentiment analysis to the entire dataset
batch_size = 64  # Adjust batch size based on memory (increase if using GPU)
sentiments_with_scores = process_in_batches(sampled_data, batch_size)

# Separate sentiment and score into two new columns
sampled_data['sentiment'] = [sentiment for sentiment, score in sentiments_with_scores]
sampled_data['sentiment_score'] = [score for sentiment, score in sentiments_with_scores]

# Save the updated dataset
output_file = "sampled_data_with_sentiment.csv"
sampled_data.to_csv(output_file, index=False)
print(f"Sentiment analysis completed. Results saved to {output_file}")

# Preview the updated DataFrame
print(sampled_data[['reviews', 'sentiment', 'sentiment_score']].head())
