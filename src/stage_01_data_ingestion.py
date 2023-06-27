import requests
import gzip
import io
import pandas as pd

def download_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception if request was unsuccessful
        return response.content
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error occurred during data download: {e}")

def unzip_data(data):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as f:
            unzipped_data = f.read()
        return unzipped_data
    except Exception as e:
        raise Exception(f"Error occurred during data unzipping: {e}")

def save_data(data, output_file):
    try:
        with open(output_file, 'wb') as f:
            f.write(data)
        print(f"Data saved: {output_file}")
    except Exception as e:
        raise Exception(f"Error occurred during data saving: {e}")

def convert_to_parquet(csv_file, parquet_file):
    try:
        df = pd.read_csv(csv_file)
        df.to_parquet(parquet_file)
        print(f"Data converted to Parquet: {parquet_file}")
    except Exception as e:
        raise Exception(f"Error occurred during Parquet conversion: {e}")

def main():
    url = "http://data.insideairbnb.com/france/ile-de-france/paris/2023-03-13/data/reviews.csv.gz"
    output_file = "data/paris_reviews.csv"
    parquet_file = "data/paris_reviews.parquet"

    try:
        # Download data
        data = download_data(url)

        # Unzip data
        unzipped_data = unzip_data(data)

        # Save data
        save_data(unzipped_data, output_file)

        # Convert to Parquet
        convert_to_parquet(output_file, parquet_file)
    except Exception as e:
        print(f"Error occurred during data processing: {e}")

if __name__ == "__main__":
    main()
