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
from tensorflow.keras.layers import Embedding, Conv1D, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Dropout, Input
from tensorflow.keras import Model
import numpy as np
import re
import string

# Try to import NLTK packages, with fallback methods if not available
try:
    import nltk
    # Try to download necessary resources, with error handling
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Initialize NLTK components
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        nltk_available = True
        print("NLTK resources successfully loaded")
    except Exception as e:
        print(f"NLTK resource download failed: {str(e)}")
        nltk_available = False
except ImportError:
    print("NLTK not available, using fallback text processing")
    nltk_available = False

# --- 1. Load your dataset ---
df = pd.read_csv("here.csv")  # Replace with your dataset's filename

try:
    # --- 2. Initialize Weaviate client and model ---
    # Use environment variables or configuration for API keys (more secure than hardcoded)
    weaviate_url = os.getenv("WEAVIATE_URL", "https://yorqaaxaqn2qspctsa0ezg.c0.us-west3.gcp.weaviate.cloud")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "6ovKNoXIRJLVbmIcBrJcTK1HsSe3AoaGlidk")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    # Replace BERT with sentence-transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # --- 3. Text preprocessing and feature extraction ---
    # Define fallback preprocessing function (no NLTK)
    def fallback_preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = ''.join([c for c in text if c not in string.punctuation])
        # Basic tokenization by splitting on whitespace
        tokens = text.split()
        # Simple stopword removal with a basic list
        basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                          'in', 'on', 'at', 'by', 'for', 'with', 'about', 'to', 'from'}
        tokens = [token for token in tokens if token not in basic_stopwords]
        return ' '.join(tokens)
    
    # Select appropriate preprocessing function based on NLTK availability
    def preprocess_text(text):
        if nltk_available:
            # Use NLTK preprocessing
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
            return ' '.join(tokens)
        else:
            # Use fallback preprocessing
            return fallback_preprocess_text(text)
    
    # Extract keywords from a text based on frequency
    def extract_keywords(text, n=10):
        # Preprocessing
        processed_text = preprocess_text(text)
        # Simple tokenization by splitting
        words = processed_text.split()
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Only consider words with length > 2
                word_freq[word] = word_freq.get(word, 0) + 1
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        # Return top n keywords
        return [word for word, freq in sorted_words[:n]]
    
    # Detect negation in a text
    def has_negation(text):
        negation_terms = ['not', 'no', 'never', "n't", 'neither', 'nor', 'cannot']
        # Simple tokenization by splitting on whitespace
        tokens = text.lower().split()
        return any(term in tokens for term in negation_terms)
    
    # Calculate semantic similarity between two texts using dot product (cosine similarity)
    def semantic_similarity(text1, text2, model):
        # Get embeddings
        emb1 = model.encode(text1)
        emb2 = model.encode(text2)
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity
    
    # Enhanced claim comparison function
    def compare_claims(claim, client, model, class_name="Mfact"):
        # Extract keywords from the claim
        claim_keywords = extract_keywords(claim)
        
        # Check for negation in the claim
        negation_present = has_negation(claim)
        
        # Get the embedding for the claim
        claim_embedding = model.encode(claim).tolist()
        
        # Query Weaviate
        response = client.collections.get(class_name).query.near_vector(
            near_vector=claim_embedding,
            limit=5,  # Increased from 3 to 5 for better coverage
            return_properties=["diseaseName", "cause", "symptoms"]
        )
        
        best_match_score = 0
        best_match_text = ""
        keyword_match_ratio = 0
        
        if response and len(response.objects) > 0:
            total_matches = 0
            
            for item in response.objects:
                properties = item.properties
                # Combine all non-null properties
                combined_claim = " ".join(filter(None, [
                    properties.get("diseaseName", ""), 
                    properties.get("cause", ""), 
                    properties.get("symptoms", "")
                ]))
                
                # Calculate semantic similarity
                sim_score = semantic_similarity(claim, combined_claim, model)
                
                # Extract keywords from the database entry
                db_keywords = extract_keywords(combined_claim)
                
                # Calculate keyword overlap
                common_keywords = set(claim_keywords).intersection(set(db_keywords))
                if claim_keywords:  # Avoid division by zero
                    keyword_ratio = len(common_keywords) / len(claim_keywords)
                else:
                    keyword_ratio = 0
                
                # Check for contextual consistency with negation
                contextual_match = True
                if negation_present:
                    # If claim has negation but database doesn't or vice versa, reduce similarity
                    if not has_negation(combined_claim):
                        sim_score *= 0.5
                        contextual_match = False
                
                # Combine the scores (similarity and keyword overlap)
                combined_score = 0.7 * sim_score + 0.3 * keyword_ratio
                
                # Track the best match
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match_text = combined_claim
                    keyword_match_ratio = keyword_ratio
                
                # Check for direct string containment (substring match)
                if combined_claim.lower() in claim.lower() or claim.lower() in combined_claim.lower():
                    # If there's a direct match, increase importance of this factor
                    total_matches += 1
            
            # Make final decision based on combined criteria
            if total_matches > 0:
                return "true"  # Strong evidence for a true claim
            elif best_match_score > 0.7:  # Threshold for high similarity
                return "true"  # High similarity suggests true claim
            elif best_match_score > 0.5 and keyword_match_ratio > 0.6:
                return "true"  # Good similarity with high keyword overlap
            else:
                return "false"  # Insufficient evidence for a true claim
        
        return "not known"  # No relevant information found

    # --- 4. Apply the comparison function to the dataset ---
    print("Applying enhanced comparison function to dataset...")
    
    # Add feature columns
    df['preprocessed_text'] = df['text'].apply(preprocess_text)
    df['has_negation'] = df['text'].apply(has_negation)
    df['keywords'] = df['text'].apply(lambda x: ", ".join(extract_keywords(x)))
    
    # Apply the enhanced comparison function
    df['predicted_label'] = df['text'].apply(lambda x: compare_claims(x, client, model))
    
    # Convert boolean False to string "false" for consistency
    df['label_str'] = df['label'].apply(lambda x: str(x).lower())
    
    # Print unique values for debugging
    print("Unique values in label column:", df['label'].unique())
    print("Unique values in converted label_str column:", df['label_str'].unique())
    print("Unique values in predicted_label column:", df['predicted_label'].unique())

    # --- 5. Evaluate the enhanced Weaviate comparison ---
    actual_classes = list(set(list(df['label_str'].unique()) + list(df['predicted_label'].unique())))
    print("Actual classes present in the data:", actual_classes)
    
    le = LabelEncoder()
    le.fit(actual_classes)
    
    print("Classes learned by the encoder:", le.classes_)
    
    df['label_encoded'] = le.transform(df['label_str'])
    df['predicted_label_encoded'] = le.transform(df['predicted_label'])

    accuracy = accuracy_score(df['label_encoded'], df['predicted_label_encoded'])
    print(f"Enhanced Weaviate Comparison Accuracy: {accuracy}")
    
    present_labels = sorted(list(set(df['label_encoded'].unique()).union(set(df['predicted_label_encoded'].unique()))))
    present_class_names = [le.classes_[i] for i in present_labels]
    
    print(classification_report(df['label_encoded'], df['predicted_label_encoded'], 
                               labels=present_labels, target_names=present_class_names))

    # --- 6. Enhanced CNN-BiLSTM Model with Attention and Context Features ---
    max_len = 100
    vocab_size = 10000

    # Tokenize and pad sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Create additional features for the model
    negation_feature = df['has_negation'].astype(int).values.reshape(-1, 1)
    
    # Create keyword match features - count of keywords from true claims vs false claims
    true_claims = df[df['label_str'] == 'true']['preprocessed_text'].str.split().explode()
    false_claims = df[df['label_str'] == 'false']['preprocessed_text'].str.split().explode()
    
    true_keywords = true_claims.value_counts().head(50).index.tolist() if len(true_claims) > 0 else []
    false_keywords = false_claims.value_counts().head(50).index.tolist() if len(false_claims) > 0 else []
    
    # Count keyword occurrences in each text
    def count_keywords(text, keyword_list):
        words = set(text.split())
        return sum(1 for keyword in keyword_list if keyword in words)
    
    df['true_keyword_count'] = df['preprocessed_text'].apply(lambda x: count_keywords(x, true_keywords))
    df['false_keyword_count'] = df['preprocessed_text'].apply(lambda x: count_keywords(x, false_keywords))
    
    keyword_features = df[['true_keyword_count', 'false_keyword_count']].values
    
    # Combine all features
    y_encoded = df['label_encoded'].values

    # Check class distribution and create synthetic examples for underrepresented classes
    classes, counts = np.unique(y_encoded, return_counts=True)
    min_class_count = 5  # Minimum number of examples per class
    
    synthetic_padded_sequences = []
    synthetic_negation_features = []
    synthetic_keyword_features = []
    synthetic_labels = []
    
    for cls_idx, count in zip(classes, counts):
        if count < min_class_count:
            class_name = le.classes_[cls_idx]
            n_synthetic = min_class_count - count
            print(f"Creating {n_synthetic} synthetic examples for class '{class_name}'")
            
            # Create synthetic examples
            for i in range(n_synthetic):
                # Create a synthetic text (just for padding purposes)
                synthetic_text = f"This is a synthetic {class_name} claim example {i+1}."
                synthetic_seq = tokenizer.texts_to_sequences([synthetic_text])
                synthetic_padded = pad_sequences(synthetic_seq, maxlen=max_len, padding='post', truncating='post')
                
                # Create synthetic features
                synthetic_neg = np.array([[0]])  # No negation
                
                # Keywords - bias toward this class 
                if class_name == 'true':
                    synthetic_kw = np.array([[5, 1]])  # High true, low false keywords
                else:
                    synthetic_kw = np.array([[1, 5]])  # Low true, high false keywords
                
                # Add to synthetic collections
                synthetic_padded_sequences.append(synthetic_padded[0])
                synthetic_negation_features.append(synthetic_neg[0])
                synthetic_keyword_features.append(synthetic_kw[0])
                synthetic_labels.append(cls_idx)
    
    # If we created any synthetic examples, add them to the dataset
    if synthetic_labels:
        padded_sequences = np.vstack([padded_sequences, np.array(synthetic_padded_sequences)])
        negation_feature = np.vstack([negation_feature, np.array(synthetic_negation_features)])
        keyword_features = np.vstack([keyword_features, np.array(synthetic_keyword_features)])
        y_encoded = np.append(y_encoded, np.array(synthetic_labels))
        
        print(f"Added {len(synthetic_labels)} synthetic examples. New class distribution:")
        for cls, count in zip(*np.unique(y_encoded, return_counts=True)):
            print(f"  Class {le.classes_[cls]}: {count} examples")

    # Split the data - REMOVING STRATIFICATION to fix the error
    indices = np.arange(len(padded_sequences))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42
        # Removed stratify=y_encoded
    )
    
    X_train_seq = padded_sequences[train_indices]
    X_test_seq = padded_sequences[test_indices]
    X_train_negation = negation_feature[train_indices]
    X_test_negation = negation_feature[test_indices]
    X_train_keywords = keyword_features[train_indices]
    X_test_keywords = keyword_features[test_indices]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    # Check if test set has all classes
    test_classes = np.unique(y_test)
    if len(test_classes) < len(le.classes_):
        print("Warning: Test set does not contain all classes. This might affect evaluation.")
        missing_classes = [le.classes_[i] for i in range(len(le.classes_)) if i not in test_classes]
        print(f"Missing classes in test set: {missing_classes}")

    # Build an enhanced model with multiple input features
    # Text input branch
    text_input = Input(shape=(max_len,), name='text_input')
    embedding = Embedding(vocab_size, 128, input_length=max_len)(text_input)
    conv = Conv1D(128, 5, activation='relu')(embedding)
    bilstm1 = Bidirectional(LSTM(64, return_sequences=True))(conv)
    bilstm2 = Bidirectional(LSTM(32))(bilstm1)
    text_features = Dense(64, activation='relu')(bilstm2)
    
    # Negation feature branch
    negation_input = Input(shape=(1,), name='negation_input')
    negation_features = Dense(8, activation='relu')(negation_input)
    
    # Keyword feature branch
    keyword_input = Input(shape=(2,), name='keyword_input')
    keyword_features = Dense(16, activation='relu')(keyword_input)
    
    # Merge all features
    merged = tf.keras.layers.concatenate([text_features, negation_features, keyword_features])
    
    # Final classification layers
    dense1 = Dense(64, activation='relu')(merged)
    dropout = Dropout(0.5)(dense1)
    output = Dense(len(le.classes_), activation='softmax')(dropout)
    
    # Create the model
    model_enhanced = Model(
        inputs=[text_input, negation_input, keyword_input],
        outputs=output
    )
    
    model_enhanced.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Training Enhanced CNN-BiLSTM model with context features...")
    history = model_enhanced.fit(
        [X_train_seq, X_train_negation, X_train_keywords],
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Predictions
    y_pred_probs = model_enhanced.predict([X_test_seq, X_test_negation, X_test_keywords])
    y_pred = y_pred_probs.argmax(axis=-1)

    # For the final classification report
    present_test_labels = sorted(list(set(y_test).union(set(y_pred))))
    present_test_class_names = [le.classes_[i] for i in present_test_labels]
    
    print("Enhanced Model Classification Report:")
    print(classification_report(y_test, y_pred, labels=present_test_labels, target_names=present_test_class_names))
    print("Enhanced Model Accuracy:", accuracy_score(y_test, y_pred))
    
    # --- 7. Analyze True vs False Performance ---
    # Check if both true and false classes exist in the data
    true_idx = list(le.classes_).index('true') if 'true' in le.classes_ else None
    false_idx = list(le.classes_).index('false') if 'false' in le.classes_ else None
    
    if true_idx is not None and false_idx is not None:
        # Get masks for true and false examples
        true_mask = (y_test == true_idx)
        false_mask = (y_test == false_idx)
        
        # Calculate class-specific accuracy
        true_accuracy = accuracy_score(y_test[true_mask], y_pred[true_mask]) if np.any(true_mask) else 0
        false_accuracy = accuracy_score(y_test[false_mask], y_pred[false_mask]) if np.any(false_mask) else 0
        
        print(f"\nClass-specific performance:")
        print(f"True claims accuracy: {true_accuracy:.4f}")
        print(f"False claims accuracy: {false_accuracy:.4f}")
        
        # Analyze examples where model was correct/incorrect
        true_correct = np.where(np.logical_and(true_mask, y_test == y_pred))[0]
        true_incorrect = np.where(np.logical_and(true_mask, y_test != y_pred))[0]
        false_correct = np.where(np.logical_and(false_mask, y_test == y_pred))[0]
        false_incorrect = np.where(np.logical_and(false_mask, y_test != y_pred))[0]
        
        print("\nExample Analysis:")
        
        # Function to get original text from test index
        def get_text_from_index(idx):
            return df.iloc[test_indices[idx]]['text']
        
        # Print some examples for analysis
        if len(true_correct) > 0:
            print("\nTrue claims correctly identified:")
            for idx in true_correct[:min(2, len(true_correct))]:
                print(f"- {get_text_from_index(idx)}")
        
        if len(true_incorrect) > 0:
            print("\nTrue claims incorrectly identified as false:")
            for idx in true_incorrect[:min(2, len(true_incorrect))]:
                print(f"- {get_text_from_index(idx)}")
        
        if len(false_correct) > 0:
            print("\nFalse claims correctly identified:")
            for idx in false_correct[:min(2, len(false_correct))]:
                print(f"- {get_text_from_index(idx)}")
        
        if len(false_incorrect) > 0:
            print("\nFalse claims incorrectly identified as true:")
            for idx in false_incorrect[:min(2, len(false_incorrect))]:
                print(f"- {get_text_from_index(idx)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # --- 8. Close the Weaviate connection properly ---
    if 'client' in locals():
        client.close()
        print("Weaviate client closed properly")