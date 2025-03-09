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

    # --- 3. Function to query Weaviate and compare claims ---
    def compare_claims(claim, client, model, class_name="Mfact"):
        claim_embedding = model.encode(claim).tolist()

        # Weaviate 4.0+ Query Syntax - no .do() at the end
        response = client.collections.get(class_name).query.near_vector(
            near_vector=claim_embedding,
            limit=3,
            return_properties=["diseaseName", "cause", "symptoms"]
        )

        if response and len(response.objects) > 0:
            for item in response.objects:
                properties = item.properties
                combined_claim = " ".join(filter(None, [
                    properties.get("diseaseName"), 
                    properties.get("cause"), 
                    properties.get("symptoms")
                ]))
                if combined_claim.lower() in claim.lower() or claim.lower() in combined_claim.lower():
                    return "true"  # Return string, not boolean
            return "false"  # Return string, not boolean
        return "not known"

    # --- 4. Apply the comparison function to the dataset ---
    print("Applying comparison function to dataset...")
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

    accuracy = accuracy_score(df['label_encoded'], df['predicted_label_encoded'])
    print(f"Weaviate Comparison Accuracy: {accuracy}")
    
    # Use only the classes that actually appear in the data for the classification report
    # Identify the labels that are actually present in the test data
    present_labels = sorted(list(set(df['label_encoded'].unique()).union(set(df['predicted_label_encoded'].unique()))))
    present_class_names = [le.classes_[i] for i in present_labels]
    
    # Now use the labels parameter to ensure consistency
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
    
    print("Training CNN-BiLSTM model...")
    model_cnn_bilstm.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=1)

    # Predictions
    y_pred_probs = model_cnn_bilstm.predict(X_test)
    y_pred = y_pred_probs.argmax(axis=-1)

    # For the final classification report, identify which classes actually appear in test data
    present_test_labels = sorted(list(set(y_test).union(set(y_pred))))
    present_test_class_names = [le.classes_[i] for i in present_test_labels]
    
    print("CNN-BiLSTM Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_test_labels, target_names=present_test_class_names))
    print("CNN-BiLSTM Accuracy:", accuracy_score(y_test, y_pred))

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # --- 7. Close the Weaviate connection properly (even if an exception occurs) ---
    if 'client' in locals():
        client.close()
        print("Weaviate client closed properly")