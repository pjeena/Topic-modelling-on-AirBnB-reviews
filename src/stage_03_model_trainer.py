import pandas as pd
import numpy as np
import glob
import os
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
import nltk
nltk.download('stopwords')



def train_bertopic(data):
    try:
        # Initiate UMAP
        umap_model = UMAP(n_neighbors=15, 
                        n_components=5, 
                        min_dist=0.0, 
                        metric='cosine', 
                        random_state=100)
        


        #NLTK English stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        airbnb_related_words = ['stay', 'airbnb', 'paris', 'would', 'time', 'apartment']
        names_and_surnames = pd.read_csv('data/names_and_surnames.csv')
        # Expand stopwords
        stopwords.extend(list(names_and_surnames['names_&_surnames']) + airbnb_related_words)


        vectorizer_model = CountVectorizer(stop_words=stopwords)
        representation_model = MaximalMarginalRelevance(diversity=0.8)

        # Initiate BERTopic
        topic_model = BERTopic(umap_model=umap_model, 
                            vectorizer_model=vectorizer_model, 
        #                      min_topic_size=200,
        #                       top_n_words=4,
                            language="multilingual",
                            calculate_probabilities=True,
                            representation_model=representation_model)



        import time
        start = time.time()


        # Run BERTopic model
        topics,_ = topic_model.fit_transform(data)

        end = time.time()
        print(end - start)

        # Return the trained model and topics
        return topic_model, topics
    
    except Exception as e:
        raise Exception(f"Error occurred during BERTopic training: {e}")
    


def main():
    parquet_file = "data/paris_reviews_preprocessed.parquet"

    try:
        # Process the Parquet file
        df = pd.read_parquet(parquet_file)
        docs = df.comments

        # Train the BERTopic model
        model, topics = train_bertopic(docs)

        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        model.save("model/model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)


    except Exception as e:
        print(f"Error occurred during data processing: {e}")


if __name__ == "__main__":
    main()