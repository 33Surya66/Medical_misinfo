import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Dropout
import numpy as np

# --- 1. Load your dataset ---
df = pd.read_csv("here.csv")  # Replace with your dataset's filename

try:
    # --- 2. Initialize Weaviate client and model ---
    weaviate_url = "https://yorqaaxaqn2qspctsa0ezg.c0.us-west3.gcp.weaviate.cloud"
    weaviate_api_key = "6ovKNoXIRJLVbmIcBrJcTK1HsSe3AoaGlidk"

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 3. Enhanced function to query Weaviate and compare claims with detailed logging ---
    def compare_claims(claim, client, model, class_name="Mfact", verbose=False):
        if verbose:
            print("\n" + "="*80)
            print(f"ANALYZING CLAIM: '{claim}'")
            print("="*80)
        
        claim_embedding = model.encode(claim).tolist()

        # Weaviate 4.0+ Query Syntax - no .do() at the end
        response = client.collections.get(class_name).query.near_vector(
            near_vector=claim_embedding,
            limit=3,
            return_properties=["diseaseName", "cause", "symptoms"]
        )

        if verbose:
            print("\nTOP MATCHES FROM VECTOR DB:")
            print("-"*60)
        
        if response and len(response.objects) > 0:
            for i, item in enumerate(response.objects):
                properties = item.properties
                combined_claim = " ".join(filter(None, [
                    properties.get("diseaseName"), 
                    properties.get("cause"), 
                    properties.get("symptoms")
                ]))
                
                if verbose:
                    print(f"Match #{i+1}:")
                    print(f"  Disease: {properties.get('diseaseName', 'N/A')}")
                    print(f"  Cause: {properties.get('cause', 'N/A')}")
                    print(f"  Symptoms: {properties.get('symptoms', 'N/A')}")
                    print(f"  Combined: '{combined_claim}'")
                    
                    # Calculate simple similarity metrics
                    claim_words = set(claim.lower().split())
                    db_words = set(combined_claim.lower().split())
                    common_words = claim_words.intersection(db_words)
                    
                    print(f"  Overlap: {len(common_words)} words")
                    if len(claim_words) > 0:
                        print(f"  Coverage: {len(common_words)/len(claim_words):.2%} of claim words")
                    
                    # Check for substring match
                    is_substring = combined_claim.lower() in claim.lower() or claim.lower() in combined_claim.lower()
                    print(f"  Substring match: {is_substring}")
                    print("-"*40)
                
                # Determine if it's a match
                if combined_claim.lower() in claim.lower() or claim.lower() in combined_claim.lower():
                    if verbose:
                        print("\nRESULT: TRUE - Found matching claim in database")
                    return "true"  # Return string, not boolean
            
            if verbose:
                print("\nRESULT: FALSE - No matching claim in database")
            return "false"  # Return string, not boolean
        
        if verbose:
            print("\nRESULT: NOT KNOWN - No relevant matches found in database")
        return "not known"
    
    # --- 4. Apply the comparison function to the dataset ---
    print("Applying comparison function to dataset...")
    
    # First, let's analyze a few examples in detail to understand the matching process
    print("\n\nDETAILED ANALYSIS OF SAMPLE CLAIMS:")
    print("="*80)
    
    # Get a sample of claims to analyze in detail (e.g., first 5)
    sample_size = min(5, len(df))
    for idx, row in df.iloc[:sample_size].iterrows():
        claim_text = row['text']
        actual_label = row['label']
        predicted = compare_claims(claim_text, client, model, verbose=True)
        print(f"Actual label: {actual_label}, Predicted: {predicted}")
        print("="*80)
    
    # Now process the whole dataset
    df['predicted_label'] = df['text'].apply(lambda x: compare_claims(x, client, model))
    
    # Convert boolean False to string "false" for consistency
    df['label_str'] = df['label'].apply(lambda x: str(x).lower())
    
    # Print unique values for debugging
    print("Unique values in label column:", df['label'].unique())
    print("Unique values in converted label_str column:", df['label_str'].unique())
    print("Unique values in predicted_label column:", df['predicted_label'].unique())

    # --- 5. Evaluate the Weaviate comparison ---
    # Identify what classes actually appear in the data
    actual_classes = list(set(list(df['label_str'].unique()) + list(df['predicted_label'].unique())))
    print("Actual classes present in the data:", actual_classes)
    
    # Encode labels using only the actual classes
    le = LabelEncoder()
    le.fit(actual_classes)
    
    print("Classes learned by the encoder:", le.classes_)
    
    df['label_encoded'] = le.transform(df['label_str'])
    df['predicted_label_encoded'] = le.transform(df['predicted_label'])

    # Create a comparison dataframe to visualize results
    print("\nSAMPLE COMPARISON OF ACTUAL VS PREDICTED LABELS:")
    print("="*80)
    comparison_df = df[['text', 'label_str', 'predicted_label']].copy()
    comparison_df['matches'] = comparison_df['label_str'] == comparison_df['predicted_label']
    print(comparison_df.head(10))
    
    # Save the comparison to CSV for further analysis
    comparison_df.to_csv("vector_comparison_results.csv", index=False)
    print("\nFull comparison saved to 'vector_comparison_results.csv'")
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['label_encoded'], df['predicted_label_encoded'])
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print class distribution
    print("\nClass Distribution:")
    class_counts = df['label_str'].value_counts()
    print(class_counts)
    
    # Overall accuracy
    accuracy = accuracy_score(df['label_encoded'], df['predicted_label_encoded'])
    print(f"\nWeaviate Comparison Accuracy: {accuracy:.4f}")
    
    # Use only the classes that actually appear in the data for the classification report
    # Identify the labels that are actually present in the test data
    present_labels = sorted(list(set(df['label_encoded'].unique()).union(set(df['predicted_label_encoded'].unique()))))
    present_class_names = [le.classes_[i] for i in present_labels]
    
    # Now use the labels parameter to ensure consistency
    print("\nClassification Report:")
    print(classification_report(df['label_encoded'], df['predicted_label_encoded'], 
                               labels=present_labels, target_names=present_class_names))

    # --- 6. CNN-BiLSTM Model ---
    max_len = 100
    vocab_size = 10000

    # Tokenize and pad sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Create proper numeric labels for the model
    y_encoded = df['label_encoded'].values

    # Check if we have enough classes for meaningful classification
    n_classes = len(le.classes_)
    if n_classes <= 1:
        print("Warning: Only one class found in the data. Classification requires at least two classes.")
        print("Adding synthetic examples for demonstration purposes...")
        
        # Create some synthetic examples for the missing class(es)
        synthetic_texts = [
            "This is a synthetic true claim for demonstration.",
            "Another synthetic true claim for the model to learn."
        ]
        synthetic_labels = ["true"] * len(synthetic_texts)
        
        # Encode and add the synthetic examples
        synthetic_sequences = tokenizer.texts_to_sequences(synthetic_texts)
        synthetic_padded = pad_sequences(synthetic_sequences, maxlen=max_len, padding='post', truncating='post')
        
        # If 'true' isn't in the encoder, we need to update it
        if "true" not in le.classes_:
            old_classes = le.classes_.tolist()
            old_classes.append("true")
            le.classes_ = np.array(old_classes)
        
        synthetic_encoded = le.transform(synthetic_labels)
        
        # Combine with original data
        padded_sequences = np.vstack([padded_sequences, synthetic_padded])
        y_encoded = np.append(y_encoded, synthetic_encoded)
        
        print(f"Added {len(synthetic_texts)} synthetic examples. New class distribution:")
        for cls, count in zip(*np.unique(y_encoded, return_counts=True)):
            print(f"  Class {le.classes_[cls]}: {count} examples")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Build the CNN-BiLSTM model - FIXED VERSION
    model_cnn_bilstm = Sequential([
        Embedding(vocab_size, 128, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        # Removed GlobalMaxPooling1D() layer to preserve temporal dimension for LSTM
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])

    model_cnn_bilstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("\nTraining CNN-BiLSTM model...")
    model_cnn_bilstm.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=1)

    # Predictions
    y_pred_probs = model_cnn_bilstm.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=-1)

    # For the final classification report, identify which classes actually appear in test data
    present_test_labels = sorted(list(set(y_test).union(set(y_pred))))
    present_test_class_names = [le.classes_[i] for i in present_test_labels]
    
    print("\nCNN-BiLSTM Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_test_labels, target_names=present_test_class_names))
    print("CNN-BiLSTM Accuracy:", accuracy_score(y_test, y_pred))
    
    # Compare CNN-BiLSTM vs Vector DB approach on test data
    print("\nCOMPARISON OF APPROACHES ON TEST DATA:")
    print("="*80)
    
    # Convert test indices back to original dataframe indices
    test_indices = []
    for i in range(len(df)):
        if i in range(len(X_test)):
            test_indices.append(i)
    
    # Extract a few test examples to show detailed comparison
    test_sample = df.iloc[test_indices[:5]]
    
    for idx, row in test_sample.iterrows():
        print(f"\nClaim: '{row['text']}'")
        print(f"True label: {row['label_str']}")
        print(f"Vector DB prediction: {row['predicted_label']}")
        
        # Get the index in the test set
        test_idx = test_indices.index(idx)
        if test_idx < len(y_pred):
            cnn_pred_idx = y_pred[test_idx]
            cnn_pred_label = le.classes_[cnn_pred_idx]
            print(f"CNN-BiLSTM prediction: {cnn_pred_label}")
            print(f"Prediction probabilities: {y_pred_probs[test_idx]}")
        
        print("-"*60)

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # --- 7. Close the Weaviate connection properly (even if an exception occurs) ---
    if 'client' in locals():
        client.close()
        print("Weaviate client closed properly")