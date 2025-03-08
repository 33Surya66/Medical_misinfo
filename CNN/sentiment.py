import pandas as pd
from textblob import TextBlob

def analyze_medical_misinformation_sentiment(csv_file, text_column):
    '''
    Performs sentiment analysis on medical misinformation text from a CSV file.

    Args:
        csv_file (str): "C:/Users/khatr/OneDrive/Documents/PBL_2/health_posts.csv"
        text_column (str): "Title"

    Returns:
        pandas.DataFrame: DataFrame with added sentiment analysis columns (polarity, subjectivity, sentiment label).
    '''
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_file}' is empty.")
        return None
    except Exception as e:
        print(f"An error occurred reading the CSV file: {e}")
        return None

    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in the CSV file.")
        return None

    def get_sentiment(text):
        try:
            analysis = TextBlob(str(text))  # Convert to string to handle potential NaN values
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity

            if polarity > 0:
                sentiment_label = "Positive"
            elif polarity < 0:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            return polarity, subjectivity, sentiment_label
        except Exception as e:
            print(f"Error processing text: {text}. Error: {e}")
            return None, None, "Error"

    sentiment_results = df[text_column].apply(get_sentiment)

    # Handle potential None returns from error case.
    df[['polarity', 'subjectivity', 'sentiment_label']] = pd.DataFrame(sentiment_results.tolist(), index=df.index)

    return df

# Example usage:
csv_file_path ="C:/Users/khatr/OneDrive/Documents/PBL_2/health_posts.csv" # Replace with your CSV file path
text_column_name = "Title"  # Replace with the name of your text column

result_df = analyze_medical_misinformation_sentiment(csv_file_path, text_column_name)

if result_df is not None:
    print(result_df.head()) # Print the first few rows of the result.
    result_df.to_csv("sentiment_analyzed_posts.csv", index=False) #save to a new csv file