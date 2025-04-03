import os
import pandas as pd
import numpy as np
from weaviate import Client
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Dropout
import re
import concurrent.futures
from functools import lru_cache
import gc
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("here.csv")  # Replace with your dataset's filename

# Initialize Weaviate client and model
weaviate_url = "https://kevljiqstmkp1adi7rofw.c0.asia-southeast1.gcp.weaviate.cloud"
weaviate_api_key = "j19hysrixS3ngxAoRW6iGrk89NrjZdKcyF9D"

client = Client(
    url=weaviate_url,
    auth_client_secret=AuthApiKey(api_key=weaviate_api_key)
)

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions with caching for efficiency
@lru_cache(maxsize=512)
def get_embedding(text):
    """Cached function to get embeddings"""
    return sentence_model.encode(text, show_progress_bar=False).tolist()

def preprocess_claim(claim):
    """Split compound claims into simpler parts for better matching"""
    if " and " in claim or ", " in claim:
        subclaims = []
        for part in re.split(r' and |, ', claim):
            if len(part.strip()) > 10:
                subclaims.append(part.strip())
        return subclaims
    return [claim]

# Vector Search with probabilistic confidence
def process_single_claim(subclaim, client, model, class_name="Mfact", verbose=False, 
                       true_threshold=0.65, false_threshold=0.80):
    """Process a single claim and return confidence scores for both true and false"""
    # Input validation
    if not isinstance(subclaim, str):
        subclaim = str(subclaim)
    
    if verbose:
        print(f"\nProcessing claim: '{subclaim}'")
    
    # Safety check - empty string
    if not subclaim or subclaim.strip() == "":
        if verbose:
            print("\nRESULT: NOT KNOWN - Empty claim")
        return "not known", 0.1, 0.1, 0.8
    
    try:
        subclaim_embedding = get_embedding(subclaim)
        
        # Updated Weaviate query to use the correct API version
        try:
            response = client.query.get(class_name, ["diseaseName", "cause", "symptoms", "measures", "cure"]) \
                .with_near_vector({"vector": subclaim_embedding}) \
                .with_limit(10) \
                .do()
            
            # Handle the response according to the query format
            result_objects = response.get('data', {}).get('Get', {}).get(class_name, [])
        except Exception as e:
            if verbose:
                print(f"Error querying Weaviate: {str(e)}")
            return "not known", 0.1, 0.1, 0.8
        
        if not result_objects or len(result_objects) == 0:
            if verbose:
                print("\nRESULT: NOT KNOWN - No matches found")
            # Return equal low probabilities for true/false and high for not_known
            return "not known", 0.1, 0.1, 0.8
        
        best_similarity = 0
        best_match = None
        supporting_evidence_count = 0
        
        for i, item in enumerate(result_objects):
            if item is None:
                continue
                
            properties = item
            # Skip problematic items
            if not isinstance(properties, dict):
                continue
                
            if any(bad_text in str(properties).lower() for bad_text in 
                  ["give me a better", "approachjh", "deteccting", "harldy", "random"]):
                continue
            
            # Safely extract text fields
            combined_text = " ".join([
                str(properties.get(field, "")) for field in 
                ["diseaseName", "cause", "symptoms", "measures", "cure"]
            ])
            
            # Skip if combined text is empty
            if not combined_text.strip():
                continue
                
            try:
                db_embedding = get_embedding(combined_text)
                
                # Use the fixed cosine_similarity function
                semantic_similarity = cosine_similarity([subclaim_embedding], [db_embedding])
                
                # Safety checks for word_overlap_ratio
                subclaim_words = set(subclaim.lower().split()) if subclaim else set()
                combined_text_words = set(combined_text.lower().split()) if combined_text else set()
                
                if not subclaim_words:
                    word_overlap_ratio = 0
                else:
                    word_overlap_ratio = len(subclaim_words & combined_text_words) / len(subclaim_words)
                
                # Check for substring matches
                substring_match = False
                for phrase in subclaim.lower().split('.'):
                    if len(phrase.strip()) > 15 and phrase.strip() in combined_text.lower():
                        substring_match = True
                        break
                
                match_score = (semantic_similarity * 0.75) + (word_overlap_ratio * 0.25)
                
                if match_score > best_similarity:
                    best_similarity = match_score
                    best_match = {
                        "text": combined_text,
                        "score": match_score,
                        "similarity": semantic_similarity,
                        "overlap": word_overlap_ratio,
                        "substring": substring_match
                    }
                
                if match_score > 0.60:
                    supporting_evidence_count += 1
            except Exception as e:
                if verbose:
                    print(f"Error processing result {i}: {str(e)}")
                continue
        
        if best_match:
            # Calculate probabilistic confidence scores
            claim_length = len(subclaim.split())
            complexity_factor = min(1.1, max(0.9, claim_length / 15))
            base_threshold = true_threshold if any(term in subclaim.lower() for term in 
                                                ['can cause', 'can lead', 'associated with', 'risk factor', 'can result']) \
                            else false_threshold
            adjusted_threshold = base_threshold * complexity_factor
            
            # Calculate sigmoid normalized confidence scores
            def sigmoid(x):
                return 1 / (1 + np.exp(-5 * (x - 0.5)))
            
            # Calculate true confidence
            true_confidence = sigmoid(best_match['score']) * 0.8 + \
                            (supporting_evidence_count / 10) * 0.2
            
            # Calculate false confidence (inverse relationship with true confidence)
            false_confidence = sigmoid(1 - best_match['score']) * 0.7 + \
                              (1 - (supporting_evidence_count / 10)) * 0.3
            
            # Normalize to ensure sum is close to 1
            not_known_confidence = max(0, 1 - (true_confidence + false_confidence))
            
            # Final normalization
            total = true_confidence + false_confidence + not_known_confidence
            true_confidence /= total
            false_confidence /= total
            not_known_confidence /= total
            
            # Determine the most likely label
            if true_confidence >= false_confidence and true_confidence >= not_known_confidence:
                result = "true"
            elif false_confidence >= true_confidence and false_confidence >= not_known_confidence:
                result = "false"
            else:
                result = "not known"
            
            if verbose:
                print(f"\nRESULT: {result.upper()}")
                print(f"Confidence scores - TRUE: {true_confidence:.4f}, FALSE: {false_confidence:.4f}, NOT KNOWN: {not_known_confidence:.4f}")
            
            return result, true_confidence, false_confidence, not_known_confidence
        else:
            if verbose:
                print("\nRESULT: NOT KNOWN - No relevant matches")
            # Return low probabilities for true/false and high for not_known
            return "not known", 0.1, 0.1, 0.8
    except Exception as e:
        if verbose:
            print(f"Unexpected error in process_single_claim: {str(e)}")
        return "not known", 0.1, 0.1, 0.8

def compare_claims(claim, client, model, class_name="Mfact", verbose=False, 
                 true_threshold=0.65, false_threshold=0.80):
    """Enhanced comparison with probabilistic confidence scores"""
    # Ensure claim is a string
    if not isinstance(claim, str):
        if verbose:
            print(f"Warning: Non-string claim converted: {type(claim)} -> str")
        claim = str(claim)
    
    # Handle empty strings
    if not claim or claim.strip() == "":
        if verbose:
            print("Warning: Empty claim received")
        return "not known", 0.1, 0.1, 0.8
    
    # Process the claim
    subclaims = preprocess_claim(claim)
    
    # Handle case where preprocessing returns no subclaims
    if not subclaims:
        if verbose:
            print("Warning: No subclaims extracted")
        return "not known", 0.1, 0.1, 0.8
    
    subclaim_results = []
    true_confidences = []
    false_confidences = []
    not_known_confidences = []
    
    for subclaim in subclaims:
        try:
            result, true_conf, false_conf, not_known_conf = process_single_claim(
                subclaim, client, model, class_name, verbose, true_threshold, false_threshold
            )
            subclaim_results.append(result)
            true_confidences.append(true_conf)
            false_confidences.append(false_conf)
            not_known_confidences.append(not_known_conf)
            
            # If any subclaim is true with high confidence, return true
            if result == "true" and true_conf > 0.7:
                return "true", true_conf, false_conf, not_known_conf
        except Exception as e:
            if verbose:
                print(f"Error processing subclaim '{subclaim}': {str(e)}")
            # Don't append failed subclaims to prevent skewing results
    
    # If no subclaims were successfully processed, return not known
    if not subclaim_results:
        return "not known", 0.1, 0.1, 0.8
    
    # Average the confidences across subclaims
    avg_true_conf = sum(true_confidences) / len(true_confidences) if true_confidences else 0.0
    avg_false_conf = sum(false_confidences) / len(false_confidences) if false_confidences else 0.0
    avg_not_known_conf = sum(not_known_confidences) / len(not_known_confidences) if not_known_confidences else 0.0
    
    # Determine final result based on highest average confidence
    if avg_true_conf >= avg_false_conf and avg_true_conf >= avg_not_known_conf:
        return "true", avg_true_conf, avg_false_conf, avg_not_known_conf
    elif avg_false_conf >= avg_true_conf and avg_false_conf >= avg_not_known_conf:
        return "false", avg_true_conf, avg_false_conf, avg_not_known_conf
    else:
        return "not known", avg_true_conf, avg_false_conf, avg_not_known_conf

def process_dataset_in_batches(df, client, model, batch_size=50, n_workers=4,
                             true_threshold=0.65, false_threshold=0.80):
    """Process dataset in batches and return predictions with confidence scores"""
    results = []
    true_confidences = []
    false_confidences = []
    not_known_confidences = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            
            # Create a dictionary to track the index for each future
            for idx in batch.index:
                text = batch.loc[idx, 'text']
                # Make sure text is a string
                if not isinstance(text, str):
                    text = str(text)
                future = executor.submit(
                    compare_claims, text, client, model, "Mfact", False, true_threshold, false_threshold
                )
                futures[future] = idx
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result, true_conf, false_conf, not_known_conf = future.result()
                    batch_results.append((idx, result, true_conf, false_conf, not_known_conf))
                except Exception as e:
                    print(f"Error at index {idx}: {str(e)}")
                    batch_results.append((idx, "not known", 0.1, 0.1, 0.8))
        
        # Sort batch results by index to maintain order
        batch_results.sort()
        results.extend([r[1] for r in batch_results])
        true_confidences.extend([r[2] for r in batch_results])
        false_confidences.extend([r[3] for r in batch_results])
        not_known_confidences.extend([r[4] for r in batch_results])
        
        # Force garbage collection
        gc.collect()
        
    return results, true_confidences, false_confidences, not_known_confidences
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        # Convert to numpy arrays if they aren't already
        a_np = np.array(a)
        b_np = np.array(b)
        
        # Flatten arrays to 1D if they're multi-dimensional
        if a_np.ndim > 1:
            a_np = a_np.flatten()
        if b_np.ndim > 1:
            b_np = b_np.flatten()
        
        # Calculate norms
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        # Handle zero norms
        if norm_a == 0 or norm_b == 0:
            return 0
        
        # Calculate dot product and return similarity
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))
    except Exception as e:
        print(f"Error in cosine_similarity: {str(e)}")
        return 0  # Return 0 similarity on error
# Main execution
# Prepare data
df['label_str'] = df['label'].apply(lambda x: str(x).lower())
actual_classes = list(set(df['label_str'].unique()))
le = LabelEncoder()
le.fit(actual_classes)
df['label_encoded'] = le.transform(df['label_str'])

# Handle rare classes
class_counts = df['label_encoded'].value_counts()
print(f"Class distribution before handling rare classes: {class_counts.to_dict()}")

# Check if we have any classes with fewer than 2 samples
rare_classes = class_counts[class_counts < 2].index

if len(rare_classes) > 0:
    print(f"Found {len(rare_classes)} class(es) with only one sample. Handling rare classes...")
    
    # Combine rare classes into an "other" category
    other_class_id = len(np.unique(df['label_encoded']))
    for rare_class in rare_classes:
        # Map the rare class name to the new ID in le.classes_
        le.classes_ = np.append(le.classes_, f"other_{rare_class}")
    
    # Map rare classes to the new "other" class
    for rare_class in rare_classes:
        df.loc[df['label_encoded'] == rare_class, 'label_encoded'] = other_class_id
        df.loc[df['label_encoded'] == other_class_id, 'label_str'] = f"other_{rare_class}"
    
    print(f"Combined rare classes into 'other' category with ID {other_class_id}")
    
    # Verify classes after handling
    class_counts_after = df['label_encoded'].value_counts()
    print(f"Class distribution after handling rare classes: {class_counts_after.to_dict()}")

# Split data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df['text'].values, df['label_encoded'].values, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

# Vector Search with enhanced confidence scores
print("Processing full dataset with Vector Search...")
vector_preds, vector_true_confs, vector_false_confs, vector_not_known_confs = process_dataset_in_batches(
    df, client, sentence_model, batch_size=50, n_workers=4, true_threshold=0.65, false_threshold=0.80
)
df['vector_pred'] = vector_preds
df['vector_true_conf'] = vector_true_confs
df['vector_false_conf'] = vector_false_confs
df['vector_not_known_conf'] = vector_not_known_confs
df['vector_pred_encoded'] = le.transform([pred if pred in le.classes_ else le.classes_[0] for pred in df['vector_pred']])

vector_db_accuracy = accuracy_score(df['label_encoded'], df['vector_pred_encoded'])
print(f"\nWeaviate Comparison Accuracy: {vector_db_accuracy:.4f}")
# Find all unique classes that appear in either the true labels or predictions
unique_classes = np.unique(np.concatenate([df['label_encoded'], df['vector_pred_encoded']]))

# Generate the classification report with only classes that appear in the data
print("\nVector DB Classification Report:")
print(classification_report(
    df['label_encoded'], 
    df['vector_pred_encoded'], 
    labels=unique_classes,  # Specify which labels to include in the report
    target_names=[le.classes_[i] for i in unique_classes]  # Use corresponding class names
))

# CNN-BiLSTM Model
max_len = 100
vocab_size = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Align padded sequences with train/val/test splits
indices = df.index
train_indices = [np.where(df['text'] == x)[0][0] for x in X_train]
val_indices = [np.where(df['text'] == x)[0][0] for x in X_val]
test_indices = [np.where(df['text'] == x)[0][0] for x in X_test]

X_train_seq = padded_sequences[train_indices]
X_val_seq = padded_sequences[val_indices]
X_test_seq = padded_sequences[test_indices]

# Ensure inputs are TensorFlow tensors
X_train_seq = tf.convert_to_tensor(X_train_seq, dtype=tf.float32)
X_val_seq = tf.convert_to_tensor(X_val_seq, dtype=tf.float32)
X_test_seq = tf.convert_to_tensor(X_test_seq, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

# Build and train CNN-BiLSTM model
model_cnn_bilstm = Sequential([
    Embedding(vocab_size, 64),
    Conv1D(64, 5, activation='relu'),
    Bidirectional(LSTM(32, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

tf.keras.mixed_precision.set_global_policy('mixed_float16')
model_cnn_bilstm.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_cnn_bilstm_model.h5', save_best_only=True, 
                                                      monitor='val_accuracy', mode='max')

# Calculate class weights for imbalanced data
class_counts = np.bincount(y_train.numpy())
print(f"Class counts in training data: {class_counts}")

# Calculate manual class weights - inverse of frequency
total_samples = len(y_train.numpy())
class_weight_dict = {}

for i in range(len(np.unique(y_train.numpy()))):
    if i < len(class_counts) and class_counts[i] > 0:
        class_weight_dict[i] = total_samples / (len(np.unique(y_train.numpy())) * class_counts[i])
    else:
        # Default weight for any class not in training data
        class_weight_dict[i] = 10.0

print(f"Class weights: {class_weight_dict}")

print("\nTraining CNN-BiLSTM model...")
history = model_cnn_bilstm.fit(
    X_train_seq, y_train,
    epochs=5, batch_size=32, 
    validation_data=(X_val_seq, y_val),
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate CNN-BiLSTM model
cnn_test_probs = model_cnn_bilstm.predict(X_test_seq, batch_size=64)
cnn_test_pred = cnn_test_probs.argmax(axis=-1)
cnn_bilstm_accuracy = accuracy_score(y_test, cnn_test_pred)
print(f"\nCNN-BiLSTM Model Accuracy: {cnn_bilstm_accuracy:.4f}")
# Find the actual classes present in the test data
test_classes = np.unique(y_test)

# Generate the classification report with only classes that appear in the test data
print("\nCNN-BiLSTM Classification Report:")
print(classification_report(
    y_test, 
    cnn_test_pred,
    labels=test_classes,  # Specify which labels to include in the report
    target_names=[le.classes_[i] for i in test_classes]  # Use corresponding class names for just those classes
))

# Model comparison and selection
# Save the models
print("\nSaving models and encoders...")

# Save CNN-BiLSTM model
model_cnn_bilstm.save('final_cnn_bilstm_model.keras')
print("Saved CNN-BiLSTM model")

# Save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved tokenizer")

# Save label encoder
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved label encoder")

# Compare both models on test set
test_texts = X_test
test_vector_preds, test_vector_true_confs, test_vector_false_confs, test_vector_not_known_confs = process_dataset_in_batches(
    pd.DataFrame({'text': test_texts}), client, sentence_model, batch_size=50, n_workers=4
)
test_cnn_probs = cnn_test_probs
test_cnn_preds = [le.classes_[p] for p in cnn_test_pred]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'text': test_texts,
    'true_label': [le.classes_[y] for y in y_test],
    'vector_pred': test_vector_preds,
    'vector_true_conf': test_vector_true_confs,
    'vector_false_conf': test_vector_false_confs,
    'vector_not_known_conf': test_vector_not_known_confs,  # Corrected to use test set values
    'cnn_bilstm_pred': test_cnn_preds
})

# Save the comparison results
comparison_df.to_csv("model_comparison_results.csv", index=False)
print("Saved comparison results to model_comparison_results.csv")

# Visualize results
print("\nGenerating visualizations...")

# Confusion matrix for Vector Search predictions
plt.figure(figsize=(10, 8))
cm = confusion_matrix(df['label_encoded'], df['vector_pred_encoded'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Vector Search Confusion Matrix')
plt.tight_layout()
plt.savefig('vector_confusion_matrix.png')
print("Saved vector search confusion matrix visualization")

# Confusion matrix for CNN-BiLSTM predictions
plt.figure(figsize=(10, 8))
cm_cnn = confusion_matrix(y_test, cnn_test_pred)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN-BiLSTM Model Confusion Matrix')
plt.tight_layout()
plt.savefig('cnn_bilstm_confusion_matrix.png')
print("Saved CNN-BiLSTM confusion matrix visualization")

# Plot confidence distribution for correct vs incorrect predictions in vector search
plt.figure(figsize=(12, 6))
correct_predictions = df['vector_pred_encoded'] == df['label_encoded']
correct_confs = [df['vector_true_conf'][i] if df['vector_pred'][i] == 'true' 
                else df['vector_false_conf'][i] if df['vector_pred'][i] == 'false'
                else df['vector_not_known_conf'][i] 
                for i in range(len(df)) if correct_predictions[i]]
incorrect_confs = [df['vector_true_conf'][i] if df['vector_pred'][i] == 'true' 
                  else df['vector_false_conf'][i] if df['vector_pred'][i] == 'false'
                  else df['vector_not_known_conf'][i] 
                  for i in range(len(df)) if not correct_predictions[i]]

sns.histplot(correct_confs, color='green', label='Correct Predictions', alpha=0.6, bins=20)
sns.histplot(incorrect_confs, color='red', label='Incorrect Predictions', alpha=0.6, bins=20)
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Confidence Distribution: Correct vs Incorrect Predictions (Vector Search)')
plt.legend()
plt.tight_layout()
plt.savefig('vector_confidence_distribution.png')
print("Saved vector confidence distribution visualization")

# Compare model performance
model_names = ['Vector Search', 'CNN-BiLSTM']
accuracies = [vector_db_accuracy, cnn_bilstm_accuracy]

plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, accuracies, color=['blue', 'orange'])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')
plt.ylim(0, 1.0)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.savefig('model_comparison.png')
print("Saved model comparison visualization")

# Create a prediction function for new data
def predict_claim(text, client=client, sentence_model=sentence_model, cnn_model=model_cnn_bilstm, 
                tokenizer=tokenizer, le=le, verbose=False):
    """Function to make predictions on new data using both vector search and CNN-BiLSTM"""
    # Vector search prediction with confidence scores
    vector_result, vector_true_conf, vector_false_conf, vector_not_known_conf = compare_claims(
        text, client, sentence_model, "Mfact", verbose=verbose
    )
    
    # CNN-BiLSTM prediction
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    cnn_probs = cnn_model.predict(padded)[0]
    cnn_pred = le.classes_[np.argmax(cnn_probs)]
    
    # Return detailed results from both models
    return {
        'text': text,
        'vector_prediction': vector_result,
        'vector_true_conf': vector_true_conf,
        'vector_false_conf': vector_false_conf,
        'vector_not_known_conf': vector_not_known_conf,
        'cnn_prediction': cnn_pred,
        'cnn_confidence': np.max(cnn_probs)
    }

# Example usage
print("\nExample predictions on test data:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    sample_text = X_test[idx]
    true_label = le.classes_[y_test[idx]]
    result = predict_claim(sample_text, verbose=True)
    
    print(f"\nText: {sample_text}")
    print(f"True Label: {true_label}")
    print(f"Vector Search Prediction: {result['vector_prediction']}")
    print(f"Vector Confidence - TRUE: {result['vector_true_conf']:.4f}, FALSE: {result['vector_false_conf']:.4f}, NOT KNOWN: {result['vector_not_known_conf']:.4f}")
    print(f"CNN-BiLSTM Prediction: {result['cnn_prediction']}")
    print(f"CNN-BiLSTM Confidence: {result['cnn_confidence']:.4f}")

# Save the final prediction function as a standalone module
def load_prediction_pipeline(model_dir='.'):
    """
    Load all the necessary models and components for making predictions
    
    Args:
        model_dir: Directory where models are saved
        
    Returns:
        Dictionary containing all necessary prediction components
    """
    try:
        # Load CNN-BiLSTM model
        cnn_model = tf.keras.models.load_model(f'{model_dir}/final_cnn_bilstm_model')
        print("Loaded CNN-BiLSTM model")
        
        # Load tokenizer
        with open(f'{model_dir}/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Loaded tokenizer")
        
        # Load label encoder
        with open(f'{model_dir}/label_encoder.pickle', 'rb') as handle:
            le = pickle.load(handle)
        print("Loaded label encoder")
        
        # Initialize sentence transformer
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence transformer model")
        
        # Initialize Weaviate client
        try:
            weaviate_url = "https://kevljiqstmkp1adi7rofw.c0.asia-southeast1.gcp.weaviate.cloud"
            weaviate_api_key = "j19hysrixS3ngxAoRW6iGrk89NrjZdKcyF9D"
            
            client = Client(
                url=weaviate_url,
                auth_client_secret=AuthApiKey(api_key=weaviate_api_key)
            )
            print("Connected to Weaviate database")
        except Exception as e:
            print(f"Could not connect to Weaviate: {str(e)}")
            client = None
        
        return {
            'cnn_model': cnn_model,
            'tokenizer': tokenizer,
            'le': le,
            'sentence_model': sentence_model,
            'client': client
        }
    
    except Exception as e:
        print(f"Error loading prediction pipeline: {str(e)}")
        return None

print("\nScript execution completed successfully!")

# Example of using the loaded pipeline
if __name__ == "__main__":
    # This will execute only when running the script directly
    pipeline = load_prediction_pipeline()
    if pipeline:
        sample_text = "COVID-19 vaccines are effective at preventing severe disease."
        result = predict_claim(
            sample_text, 
            client=pipeline['client'], 
            sentence_model=pipeline['sentence_model'], 
            cnn_model=pipeline['cnn_model'], 
            tokenizer=pipeline['tokenizer'], 
            le=pipeline['le'],
            verbose=True
        )
        
        print("\nPrediction for new sample text:")
        print(f"Text: {result['text']}")
        print(f"Vector Search Prediction: {result['vector_prediction']}")
        print(f"Vector Confidence - TRUE: {result['vector_true_conf']:.4f}, " + 
              f"FALSE: {result['vector_false_conf']:.4f}, " + 
              f"NOT KNOWN: {result['vector_not_known_conf']:.4f}")
        print(f"CNN-BiLSTM Prediction: {result['cnn_prediction']}")
        print(f"CNN-BiLSTM Confidence: {result['cnn_confidence']:.4f}")