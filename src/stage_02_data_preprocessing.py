import pandas as pd
import re
import html


def remove_html_tags(text):
    """Remove html tags from a text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_urls(text):
    """Remove urls from a string"""
    return re.sub(r'http\S+', '', text, flags=re.MULTILINE)


def lowercase(text):
    """Lowercase text"""
    return text.lower()


def preprocess_text(text):

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove URLs
    text = remove_urls(text)

    # Convert text to lowercase
    text = text.lower()

    return text


def process_parquet_file(input_parquet_file, output_parquet_file):
    try:
        # Read the Parquet file into a DataFrame
        df = pd.read_parquet(input_parquet_file)

        # Apply preprocessing to the text column
        df['comments'] = df['comments'].astype(str)
        df['comments'] = df['comments'].apply(preprocess_text)

        # Convert date to datetime and extract year. We also consider reviews from 2013-2023 with 5000 reviews from each year.
        df['date']= pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df = df[df['year'].isin([2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023])].reset_index(drop=True)
        df = df.groupby('year').apply(lambda x: x.sample(5000,random_state=100)).reset_index(drop=True)

        # Save the processed DataFrame as a new Parquet file
        df.to_parquet(output_parquet_file, index=False)
        print(f"Processed data saved as Parquet file: {output_parquet_file}")

    except Exception as e:
        raise Exception(f"Error occurred during Parquet file processing: {e}")
    

def main():
    input_parquet_file = "data/paris_reviews.parquet"
    output_parquet_file = "data/paris_reviews_preprocessed.parquet"

    try:
        # Process the Parquet file
        process_parquet_file(input_parquet_file, output_parquet_file)
    except Exception as e:
        print(f"Error occurred during data processing: {e}")


if __name__ == "__main__":
    main()
