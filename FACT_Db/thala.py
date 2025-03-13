import os
import pandas as pd
import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Bidirectional, LSTM, Dense, GlobalMaxPooling1D, Dropout
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import concurrent.futures
from functools import lru_cache
import gc
from sklearn.utils.class_weight import compute_class_weight
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 3. Helper functions with caching for efficiency ---
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

    # --- 4. Vector Search (with confidence) ---
    def process_single_claim(subclaim, client, model, class_name="Mfact", verbose=False, 
                            true_threshold=0.65, false_threshold=0.80):
        """Process a single claim or subclaim with improved precision and return confidence"""
        if verbose:
            print(f"\nProcessing claim: '{subclaim}'")
        
        subclaim_embedding = get_embedding(subclaim)
        response = client.collections.get(class_name).query.near_vector(
            near_vector=subclaim_embedding,
            limit=10,
            return_properties=["diseaseName", "cause", "symptoms", "measures", "cure"]
        )
        
        if not response or len(response.objects) == 0:
            if verbose:
                print("\nRESULT: NOT KNOWN - No matches found")
            return "not known", 0.0
        
        best_similarity = 0
        best_match = None
        supporting_evidence_count = 0
        
        for i, item in enumerate(response.objects):
            properties = item.properties
            if any(bad_text in str(properties).lower() for bad_text in 
                   ["give me a better", "approachjh", "deteccting", "harldy", "random"]):
                continue
            
            combined_text = " ".join([properties.get(field, "") for field in 
                                     ["diseaseName", "cause", "symptoms", "measures", "cure"]])
            db_embedding = get_embedding(combined_text)
            semantic_similarity = cosine_similarity([subclaim_embedding], [db_embedding])[0][0]
            word_overlap_ratio = len(set(subclaim.lower().split()) & set(combined_text.lower().split())) / \
                                len(set(subclaim.lower().split())) if subclaim.split() else 0
            substring_match = any(phrase.strip() in combined_text.lower() 
                                 for phrase in subclaim.lower().split('.') if len(phrase.strip()) > 15)
            
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
        
        if best_match:
            claim_length = len(subclaim.split())
            complexity_factor = min(1.1, max(0.9, claim_length / 15))
            base_threshold = true_threshold if any(term in subclaim.lower() for term in 
                                                  ['can cause', 'can lead', 'associated with', 'risk factor', 'can result']) \
                            else false_threshold
            adjusted_threshold = base_threshold * complexity_factor
            
            is_true = (
                (best_match['score'] >= adjusted_threshold and supporting_evidence_count >= 2) or
                (best_match['similarity'] >= 0.85 and best_match['substring']) or
                (best_match['overlap'] >= 0.6 and supporting_evidence_count >= 3)
            )
            
            confidence = best_match['score'] if is_true else (1 - best_match['score'])
            result = "true" if is_true else "false"
            
            if verbose:
                print(f"\nRESULT: {result.upper()} (confidence: {confidence:.4f})")
            return result, confidence
        else:
            if verbose:
                print("\nRESULT: NOT KNOWN - No relevant matches")
            return "not known", 0.0

    def compare_claims(claim, client, model, class_name="Mfact", verbose=False, 
                      true_threshold=0.65, false_threshold=0.80):
        """Enhanced comparison with confidence"""
        subclaims = preprocess_claim(claim)
        subclaim_results = []
        confidences = []
        
        for subclaim in subclaims:
            result, confidence = process_single_claim(
                subclaim, client, model, class_name, verbose, true_threshold, false_threshold
            )
            subclaim_results.append(result)
            confidences.append(confidence)
            
            if result == "true":
                return "true", confidence
        
        if all(result == "false" for result in subclaim_results):
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return "false", avg_confidence
        else:
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return "not known", avg_confidence

    def process_dataset_in_batches(df, client, model, batch_size=50, n_workers=4,
                                  true_threshold=0.65, false_threshold=0.80):
        """Process dataset in batches and return predictions with confidences"""
        results = []
        confidences = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
            
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        compare_claims, text, client, model, "Mfact", False, true_threshold, false_threshold
                    ): idx 
                    for idx, text in zip(batch.index, batch['text'])
                }
                
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        result, confidence = future.result()
                        batch_results.append((idx, result, confidence))
                    except Exception as e:
                        print(f"Error at index {idx}: {str(e)}")
                        batch_results.append((idx, "not known", 0.0))
            
            batch_results.sort()
            results.extend([r[1] for r in batch_results])
            confidences.extend([r[2] for r in batch_results])
            gc.collect()
            
        return results, confidences

    # --- 5. Ensemble function ---
    def ensemble_predict(text, vector_result, vector_conf, cnn_probs, bert_probs, le, 
                         weights=(0.3, 0.4, 0.3), confidence_threshold=0.7):
        """Combine predictions from vector search, CNN-BiLSTM, and BERT with weights"""
        vector_w, cnn_w, bert_w = weights
        
        cnn_pred_idx = np.argmax(cnn_probs)
        cnn_conf = cnn_probs[cnn_pred_idx]
        bert_pred_idx = np.argmax(bert_probs)
        bert_conf = bert_probs[bert_pred_idx]
        
        if vector_result == "not known" and cnn_conf < confidence_threshold and bert_conf < confidence_threshold:
            return "not known"
        
        # Convert predictions to scores
        vector_score = {"true": vector_conf if vector_result == "true" else 0,
                       "false": vector_conf if vector_result == "false" else 0}
        cnn_score = {"true": cnn_probs[le.transform(["true"])[0]] if "true" in le.classes_ else 0,
                    "false": cnn_probs[le.transform(["false"])[0]] if "false" in le.classes_ else 0}
        bert_score = {"true": bert_probs[le.transform(["true"])[0]] if "true" in le.classes_ else 0,
                     "false": bert_probs[le.transform(["false"])[0]] if "false" in le.classes_ else 0}
        
        # Weighted scores
        final_score_true = (vector_w * vector_score["true"]) + (cnn_w * cnn_score["true"]) + (bert_w * bert_score["true"])
        final_score_false = (vector_w * vector_score["false"]) + (cnn_w * cnn_score["false"]) + (bert_w * bert_score["false"])
        
        if final_score_true > final_score_false and final_score_true >= confidence_threshold:
            return "true"
        elif final_score_false > final_score_true and final_score_false >= confidence_threshold:
            return "false"
        else:
            return "not known"

    # --- 6. Main execution ---
    # Prepare data
    df['label_str'] = df['label'].apply(lambda x: str(x).lower())
    actual_classes = list(set(df['label_str'].unique()))
    le = LabelEncoder()
    le.fit(actual_classes)
    df['label_encoded'] = le.transform(df['label_str'])

    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['text'].values, df['label_encoded'].values, test_size=0.2, random_state=42, stratify=df['label_encoded']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # --- 7. Vector Search ---
    print("Processing full dataset with Vector Search...")
    vector_preds, vector_confs = process_dataset_in_batches(
        df, client, sentence_model, batch_size=50, n_workers=4, true_threshold=0.65, false_threshold=0.80
    )
    df['vector_pred'] = vector_preds
    df['vector_conf'] = vector_confs
    df['vector_pred_encoded'] = le.transform(df['vector_pred'].replace("not known", le.classes_[0]))

    vector_db_accuracy = accuracy_score(df['label_encoded'], df['vector_pred_encoded'])
    print(f"\nWeaviate Comparison Accuracy: {vector_db_accuracy:.4f}")
    print("\nVector DB Classification Report:")
    print(classification_report(df['label_encoded'], df['vector_pred_encoded'], target_names=le.classes_))

    # --- 8. CNN-BiLSTM Model ---
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

    model_cnn_bilstm = Sequential([
        Embedding(vocab_size, 64),  # Removed input_length
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

    # Compute class weights (fixing the issue with classes having only 1 sample)
    class_counts = np.bincount(y_train.numpy())
    if np.min(class_counts) < 2:
        print("Warning: Some classes have fewer than 2 samples. Using manual class weights.")
        # Calculate manual class weights - inverse of frequency
        total_samples = len(y_train.numpy())
        class_weights = {i: total_samples / (len(np.unique(y_train.numpy())) * count) 
                        for i, count in enumerate(class_counts) if count > 0}
        # If any class has 0 samples, assign a default high weight
        for i in range(len(np.unique(y_train.numpy()))):
            if i not in class_weights:
                class_weights[i] = 10.0  # High weight for underrepresented class
        class_weight_dict = class_weights
    else:
        # Original computation if all classes have at least 2 samples
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
        class_weight_dict = dict(enumerate(class_weights))

    print("\nTraining CNN-BiLSTM model...")
    history = model_cnn_bilstm.fit(
        X_train_seq, y_train,
        epochs=5, batch_size=32, 
        validation_data=(X_val_seq, y_val),
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, model_checkpoint]
    )

    cnn_test_probs = model_cnn_bilstm.predict(X_test_seq, batch_size=64)
    cnn_test_pred = cnn_test_probs.argmax(axis=-1)
    cnn_bilstm_accuracy = accuracy_score(y_test, cnn_test_pred)
    print(f"\nCNN-BiLSTM Model Accuracy: {cnn_bilstm_accuracy:.4f}")
    print("\nCNN-BiLSTM Classification Report:")
    print(classification_report(y_test, cnn_test_pred, target_names=le.classes_))

    # --- 9. BERT Model ---
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

    def encode_texts(texts, tokenizer, max_len=128):
        return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors='tf')

    X_train_bert = encode_texts(X_train, bert_tokenizer)
    X_val_bert = encode_texts(X_val, bert_tokenizer)
    X_test_bert = encode_texts(X_test, bert_tokenizer)

    bert_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("\nTraining BERT model...")
    bert_history = bert_model.fit(
        X_train_bert, y_train,
        validation_data=(X_val_bert, y_val),
        epochs=3, batch_size=16,
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)]
    )

    bert_test_probs = tf.nn.softmax(bert_model.predict(X_test_bert, batch_size=16).logits, axis=-1).numpy()
    bert_test_pred = bert_test_probs.argmax(axis=-1)
    bert_accuracy = accuracy_score(y_test, bert_test_pred)
    print(f"\nBERT Model Accuracy: {bert_accuracy:.4f}")
    print("\nBERT Classification Report:")
    print(classification_report(y_test, bert_test_pred, target_names=le.classes_))

    # --- 10. Tune ensemble weights on validation set ---
    print("\nTuning ensemble weights on validation set...")
    val_vector_preds, val_vector_confs = process_dataset_in_batches(
        pd.DataFrame({'text': X_val}), client, sentence_model, batch_size=50, n_workers=4
    )
    val_cnn_probs = model_cnn_bilstm.predict(X_val_seq, batch_size=64)
    val_bert_probs = tf.nn.softmax(bert_model.predict(X_val_bert, batch_size=16).logits, axis=-1).numpy()

    best_accuracy = 0
    best_weights = (0.33, 0.33, 0.34)
    for v_w in np.arange(0.1, 0.8, 0.1):
        for c_w in np.arange(0.1, 0.8 - v_w, 0.1):
            b_w = 1.0 - v_w - c_w
            ensemble_preds = [
                ensemble_predict(text, v_pred, v_conf, c_prob, b_prob, le, weights=(v_w, c_w, b_w))
                for text, v_pred, v_conf, c_prob, b_prob in zip(X_val, val_vector_preds, val_vector_confs, val_cnn_probs, val_bert_probs)
            ]
            ensemble_encoded = le.transform([p if p in le.classes_ else le.classes_[0] for p in ensemble_preds])
            acc = accuracy_score(y_val, ensemble_encoded)
            if acc > best_accuracy:
                best_accuracy = acc
                best_weights = (v_w, c_w, b_w)
    
    print(f"Best weights - Vector: {best_weights[0]:.2f}, CNN-BiLSTM: {best_weights[1]:.2f}, BERT: {best_weights[2]:.2f}")
    print(f"Validation accuracy with best weights: {best_accuracy:.4f}")

    # --- 11. Apply ensemble on test set ---
    test_vector_preds, test_vector_confs = process_dataset_in_batches(
        pd.DataFrame({'text': X_test}), client, sentence_model, batch_size=50, n_workers=4
    )
    test_cnn_probs = cnn_test_probs
    test_bert_probs = bert_test_probs

    ensemble_test_preds = [
        ensemble_predict(text, v_pred, v_conf, c_prob, b_prob, le, weights=best_weights)
        for text, v_pred, v_conf, c_prob, b_prob in zip(X_test, test_vector_preds, test_vector_confs, test_cnn_probs, test_bert_probs)
    ]
    ensemble_test_encoded = le.transform([p if p in le.classes_ else le.classes_[0] for p in ensemble_test_preds])
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_test_encoded)
    print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")
    print("\nEnsemble Classification Report:")
    ensemble_report = classification_report(y_test, ensemble_test_encoded, target_names=le.classes_)
    print(ensemble_report)

    # --- 12. Save results ---
    comparison_df = pd.DataFrame({
        'text': X_test,
        'true_label': [le.classes_[y] for y in y_test],
        'vector_pred': test_vector_preds,
        'cnn_bilstm_pred': [le.classes_[p] for p in cnn_test_pred],
        'bert_pred': [le.classes_[p] for p in bert_test_pred],
        'ensemble_pred': ensemble_test_preds
    })
    comparison_df.to_csv('multimodal_comparison_results.csv', index=False)

    with open('ensemble_report.txt', 'w') as f:
        f.write(f"Ensemble Accuracy: {ensemble_accuracy:.4f}\n")
        f.write(f"Vector Weight: {best_weights[0]:.2f}, CNN-BiLSTM Weight: {best_weights[1]:.2f}, BERT Weight: {best_weights[2]:.2f}\n\n")
        f.write(ensemble_report)

    # --- 13. Plot comparison ---
    plt.figure(figsize=(10, 6))
    accuracies = [vector_db_accuracy, cnn_bilstm_accuracy, bert_accuracy, ensemble_accuracy]
    plt.bar(['Vector DB', 'CNN-BiLSTM', 'BERT', 'Ensemble'], accuracies, color=['blue', 'green', 'red', 'orange'])
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig('multimodal_accuracy_comparison.png')
    plt.close()

    print("\nAnalysis completed. Results saved.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    
finally:
    if 'client' in locals():
        client.close()
    gc.collect()
    print("\nScript execution completed.")