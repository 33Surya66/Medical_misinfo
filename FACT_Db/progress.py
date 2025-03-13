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
from sklearn.metrics.pairwise import cosine_similarity
import re
import concurrent.futures
from functools import lru_cache
import gc
from sklearn.utils.class_weight import compute_class_weight
import pickle

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

    # --- 3. Helper functions with caching for efficiency ---
    @lru_cache(maxsize=512)
    def get_embedding(text):
        """Cached function to get embeddings"""
        return model.encode(text).tolist()

    def extract_key_statements(text, max_statements=5):
        """
        Extract key statements from text to improve comparison
        """
        statements = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        statements.sort(key=len, reverse=True)
        return statements[:max_statements]
    
    def preprocess_claim(claim):
        """
        Split compound claims into simpler parts for better matching
        """
        # Split compound claims
        if " and " in claim or ", " in claim:
            subclaims = []
            for part in re.split(r' and |, ', claim):
                if len(part.strip()) > 10:  # Only consider meaningful parts
                    subclaims.append(part.strip())
            return subclaims
        return [claim]

    # --- 4. Improved function to process a single subclaim ---
    def process_single_claim(subclaim, client, model, class_name="Mfact", verbose=False, 
                           true_threshold=0.55, false_threshold=0.70):
        """Process a single claim or subclaim"""
        if verbose:
            print(f"\nProcessing claim: '{subclaim}'")
            
        # Use cached embedding function
        subclaim_embedding = get_embedding(subclaim)
        
        response = client.collections.get(class_name).query.near_vector(
            near_vector=subclaim_embedding,
            limit=5,
            return_properties=["diseaseName", "cause", "symptoms", "measures", "cure"] 
        )
        
        if verbose:
            print("\nTOP MATCHES FROM VECTOR DB:")
            print("-"*60)
            
        if response and len(response.objects) > 0:
            best_similarity = 0
            best_match = None
            for i, item in enumerate(response.objects):
                properties = item.properties
                
                # Skip entries that contain non-medical content
                if any(bad_text in str(properties) for bad_text in 
                       ["give me a better", "approachjh", "deteccting", "harldy"]):
                    continue
                    
                fields_to_combine = ["diseaseName", "cause", "symptoms", "measures", "cure"]
                combined_text_parts = []
                for field in fields_to_combine:
                    if properties.get(field):
                        combined_text_parts.append(properties.get(field))
                
                combined_claim = " ".join(combined_text_parts)
                
                if verbose:
                    print(f"Match #{i+1}:")
                    for field in fields_to_combine:
                        if properties.get(field):
                            print(f"  {field}: {properties.get(field, 'N/A')[:100]}...")
                    
                # Reuse the cached embedding function
                db_embedding = get_embedding(combined_claim)
                subclaim_embedding_normalized = get_embedding(subclaim)
                
                semantic_similarity = cosine_similarity(
                    [subclaim_embedding_normalized], 
                    [db_embedding]
                )[0][0]
                
                subclaim_words = set(subclaim.lower().split())
                db_words = set(combined_claim.lower().split())
                common_words = subclaim_words.intersection(db_words)
                word_overlap_ratio = len(common_words) / len(subclaim_words) if len(subclaim_words) > 0 else 0
                
                # Check for matching phrases
                substring_match = False
                subclaim_phrases = [phrase.strip() for phrase in subclaim.lower().split('.') if len(phrase.strip()) > 10]
                for phrase in subclaim_phrases:
                    if phrase in combined_claim.lower():
                        substring_match = True
                        break
                        
                # Enhanced match score calculation with more weight on semantic similarity
                match_score = (semantic_similarity * 0.7) + (word_overlap_ratio * 0.3)
                
                if verbose:
                    print(f"  Semantic similarity: {semantic_similarity:.4f}")
                    print(f"  Word overlap: {len(common_words)} words ({word_overlap_ratio:.2%} of claim)")
                    print(f"  Key phrase match: {substring_match}")
                    print(f"  Combined match score: {match_score:.4f}")
                    print("-"*40)
                
                # Track the best match
                if match_score > best_similarity:
                    best_similarity = match_score
                    best_match = {
                        "text": combined_claim,
                        "score": match_score,
                        "similarity": semantic_similarity,
                        "overlap": word_overlap_ratio,
                        "substring": substring_match
                    }
            
            # Decision logic based on best match with conditional thresholds
            if best_match:
                if verbose:
                    print(f"\nBEST MATCH (score: {best_match['score']:.4f}):")
                    print(f"Text: {best_match['text'][:150]}...")
                
                # Use different thresholds for different claim types with adaptive adjustment
                claim_length = len(subclaim.split())
                complexity_factor = min(1.2, max(0.8, claim_length / 15))  # Adjust based on claim complexity
                
                # Lower threshold for "can lead to", "associated with" type claims
                if any(term in subclaim.lower() for term in 
                      ['can cause', 'can lead', 'associated with', 'risk factor', 'can result']):
                    base_threshold = true_threshold
                else:
                    base_threshold = false_threshold
                    
                # Apply the complexity adjustment
                adjusted_threshold = base_threshold * complexity_factor
                
                # Determine if it's a true claim based on combined evidence and appropriate threshold
                is_true = (
                    best_match['score'] >= adjusted_threshold or
                    (best_match['similarity'] >= 0.8) or
                    (best_match['substring'] and best_match['overlap'] >= 0.5)
                )
                
                if is_true:
                    if verbose:
                        print(f"\nRESULT: TRUE - Sufficient evidence found in database (threshold: {adjusted_threshold:.3f})")
                    return "true"
                else:
                    if verbose:
                        print(f"\nRESULT: FALSE - Insufficient evidence in database (threshold: {adjusted_threshold:.3f})")
                    return "false"
            else:
                if verbose:
                    print("\nRESULT: NOT KNOWN - No relevant matches found in database")
                return "not known"
        else:
            if verbose:
                print("\nRESULT: NOT KNOWN - No relevant matches found in database")
            return "not known"

    # --- 5. Optimized function to compare claims with early stopping ---
    def compare_claims(claim, client, model, class_name="Mfact", verbose=False, 
                      true_threshold=0.55, false_threshold=0.70):
        """
        Enhanced comparison with claim preprocessing, dynamic thresholds, and early stopping
        """
        if verbose:
            print("\n" + "="*80)
            print(f"ANALYZING CLAIM: '{claim}'")
            print("="*80)
        
        # Process potentially compound claims
        subclaims = preprocess_claim(claim)
        
        # For single claims, process directly
        if len(subclaims) == 1:
            return process_single_claim(
                subclaims[0], client, model, class_name, verbose, true_threshold, false_threshold
            )
        
        # For compound claims, process with early stopping
        subclaim_results = []
        for subclaim in subclaims:
            if verbose and len(subclaims) > 1:
                print(f"\nProcessing subclaim: '{subclaim}'")
                
            result = process_single_claim(
                subclaim, client, model, class_name, verbose, true_threshold, false_threshold
            )
            subclaim_results.append(result)
            
            # Early stopping - if any subclaim is true, the whole claim is true
            if result == "true":
                if verbose:
                    print(f"\nFINAL RESULT: TRUE - At least one part of the claim is supported")
                return "true"
        
        # If we get here, no subclaim was true
        if all(result == "false" for result in subclaim_results):
            if verbose:
                print(f"\nFINAL RESULT: FALSE - All parts of the claim lack support")
            return "false"
        else:
            if verbose:
                print(f"\nFINAL RESULT: NOT KNOWN - Some parts of the claim couldn't be verified")
            return "not known"
    
    # --- 6. Batch processing function for the dataset ---
    def process_dataset_in_batches(df, client, model, batch_size=50, n_workers=4,
                                  true_threshold=0.55, false_threshold=0.70):
        """Process the dataset in batches using parallel execution"""
        results = []
        
        # Process the dataset in batches to manage memory
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
            
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compare_claims, text, client, model, 
                        "Mfact", False, true_threshold, false_threshold
                    ): idx 
                    for idx, text in zip(batch.index, batch['text'])
                }
                
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        batch_results.append((idx, result))
                    except Exception as e:
                        print(f"Error processing claim at index {idx}: {str(e)}")
                        batch_results.append((idx, "not known"))
            
            # Sort by index and add to results
            batch_results.sort()
            results.extend([r[1] for r in batch_results])
            
            # Force garbage collection after batch
            gc.collect()
            
        return results
    
    # --- 7. Apply the comparison function to sample data ---
    print("Testing on sample claims...")
    
    sample_size = min(5, len(df))
    for idx, row in df.iloc[:sample_size].iterrows():
        claim_text = row['text']
        actual_label = row['label']
        predicted = compare_claims(claim_text, client, model, verbose=True, 
                                  true_threshold=0.55, false_threshold=0.70)
        print(f"Actual label: {actual_label}, Predicted: {predicted}")
        print("="*80)
    
    # --- 8. Process the entire dataset with optimized function ---
    print("\nProcessing full dataset...")
    predicted_labels = process_dataset_in_batches(
        df, client, model, batch_size=50, n_workers=4, 
        true_threshold=0.55, false_threshold=0.70
    )
    
    # Add predictions to dataframe
    df['predicted_label'] = predicted_labels
    
    # Convert labels for consistency
    df['label_str'] = df['label'].apply(lambda x: str(x).lower())
    
    # Evaluate results
    actual_classes = list(set(list(df['label_str'].unique()) + list(df['predicted_label'].unique())))
    le = LabelEncoder()
    le.fit(actual_classes)
    
    df['label_encoded'] = le.transform(df['label_str'])
    df['predicted_label_encoded'] = le.transform(df['predicted_label'])
    
    # Create a comparison dataframe
    comparison_df = df[['text', 'label_str', 'predicted_label']].copy()
    comparison_df['matches'] = comparison_df['label_str'] == comparison_df['predicted_label']
    comparison_df.to_csv("vector_comparison_results.csv", index=False)
    
    # Calculate accuracy
    accuracy = accuracy_score(df['label_encoded'], df['predicted_label_encoded'])
    print(f"\nWeaviate Comparison Accuracy: {accuracy:.4f}")
    
    # Classification report
    present_labels = sorted(list(set(df['label_encoded'].unique()).union(set(df['predicted_label_encoded'].unique()))))
    present_class_names = [le.classes_[i] for i in present_labels]
    print("\nClassification Report:")
    print(classification_report(df['label_encoded'], df['predicted_label_encoded'], 
                              labels=present_labels, target_names=present_class_names))
    
    # --- 9. CNN-BiLSTM Model with optimized architecture --- 
    max_len = 100
    vocab_size = 10000

    # Tokenize and pad sequences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    # Create proper numeric labels
    y_encoded = df['label_encoded'].values
    
    # Handle class imbalance using class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Build optimized CNN-BiLSTM model
    model_cnn_bilstm = Sequential([
        Embedding(vocab_size, 64, input_length=max_len),  # Reduced embedding dimensions
        Conv1D(64, 5, activation='relu'),  # Fewer filters
        Bidirectional(LSTM(32, return_sequences=True)),  # Smaller LSTM units
        GlobalMaxPooling1D(),  # Replace second LSTM with pooling
        Dense(32, activation='relu'),  # Smaller dense layer
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])

    # Use mixed precision for faster training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model_cnn_bilstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, restore_best_weights=True
    )
    
    print("\nTraining CNN-BiLSTM model...")
    model_cnn_bilstm.fit(
        X_train, y_train, 
        epochs=5, 
        batch_size=32,  # Explicit batch size
        validation_split=0.2, 
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    # Predictions
    y_pred_probs = model_cnn_bilstm.predict(X_test, batch_size=64)  # Larger batch for inference
    y_pred = y_pred_probs.argmax(axis=-1)
    
    # --- 10. Optimized ensemble prediction function ---
    def predict_with_ensemble_batch(texts, batch_size=32, vector_weight=0.6):
        """
        Make batch predictions using weighted ensemble of vector DB and CNN-BiLSTM
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Get CNN-BiLSTM predictions for the batch
            sequences = tokenizer.texts_to_sequences(batch_texts)
            padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            cnn_pred_probs = model_cnn_bilstm.predict(padded, batch_size=batch_size)
            cnn_pred_encoded = cnn_pred_probs.argmax(axis=1)
            
            # Get vector DB predictions
            vector_predictions = []
            for text in batch_texts:
                vector_predictions.append(compare_claims(
                    text, client, model, 
                    true_threshold=0.55, false_threshold=0.70
                ))
            
            vector_pred_encoded = le.transform(vector_predictions)
            
            # Combine predictions
            batch_results = []
            for j in range(len(batch_texts)):
                # If predictions agree, use that
                if vector_pred_encoded[j] == cnn_pred_encoded[j]:
                    batch_results.append(le.classes_[vector_pred_encoded[j]])
                    continue
                
                # Class-specific weighting
                vector_prediction = vector_predictions[j]
                if vector_prediction == "true":
                    adjusted_vector_weight = 0.55  # Less weight to vector DB for true claims
                else:
                    adjusted_vector_weight = 0.75  # Trust vector DB more for false claims
                    
                # Estimated confidence values
                vector_confidence = 0.85 if vector_prediction == "false" else 0.65
                cnn_confidence = cnn_pred_probs[j][cnn_pred_encoded[j]]
                
                # Weight the predictions
                vector_weight_adjusted = adjusted_vector_weight * vector_confidence
                cnn_weight_adjusted = (1 - adjusted_vector_weight) * cnn_confidence
                
                # Choose higher weighted confidence
                if vector_weight_adjusted >= cnn_weight_adjusted:
                    batch_results.append(le.classes_[vector_pred_encoded[j]])
                else:
                    batch_results.append(le.classes_[cnn_pred_encoded[j]])
                    
            results.extend(batch_results)
            
        return results
    
    # --- 11. Test the improved ensemble ---
    print("\nTESTING OPTIMIZED ENSEMBLE ON SAMPLE DATA:")
    print("="*80)
    
    # Take a sample of test data
    test_sample = df.sample(min(10, len(df)))
    test_texts = test_sample['text'].tolist()
    
    # Get ensemble predictions for the sample
    ensemble_predictions = predict_with_ensemble_batch(test_texts)
    
    # Display results
    for i, (idx, row) in enumerate(test_sample.iterrows()):
        print(f"\nClaim: '{row['text']}'")
        print(f"True label: {row['label_str']}")
        
        # Get vector DB prediction
        vector_pred = compare_claims(
            row['text'], client, model, 
            true_threshold=0.55, false_threshold=0.70
        )
        print(f"Vector DB prediction: {vector_pred}")
        
        # Show ensemble prediction
        print(f"Ensemble prediction: {ensemble_predictions[i]}")
        print("-"*60)
    
    # --- 12. Final evaluation with ensemble ---
    print("\nPerforming final evaluation with ensemble...")
    
    # Get ensemble predictions for test set
    test_texts = df.iloc[X_test.shape[0]:]['text'].tolist()
    ensemble_test_predictions = predict_with_ensemble_batch(test_texts)
    
    # Convert to encoded form
    ensemble_pred_encoded = le.transform(ensemble_test_predictions)
    
    # Calculate accuracy
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred_encoded)
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
    
    # Classification report
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred_encoded, 
                              labels=present_labels, target_names=present_class_names))
    
    # --- 13. Save the optimized models ---
    print("\nSaving trained models and encoders...")
    
    # Save CNN-BiLSTM model
    model_cnn_bilstm.save("optimized_cnn_bilstm_model")
    
    # Save label encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # Save tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    
    print("All models and data processors saved successfully!")
    print("Optimization completed!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    
finally:
    # Close Weaviate connection
    if 'client' in locals():
        client.close()
        print("Weaviate client closed properly")