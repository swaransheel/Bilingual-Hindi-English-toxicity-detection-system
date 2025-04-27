from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def load_test_data(csv_path):
    # Load test data
    df = pd.read_csv(csv_path)
    
    # Filter test partition
    test_df = df[df['partition'] == 'devtest']
    if len(test_df) == 0:
        raise ValueError("No test data found in the CSV file")
        
    # Keep only definite labels
    test_df = test_df[test_df['contains_toxicity'].isin(['Yes','No'])]
    if len(test_df) == 0:
        raise ValueError("No valid labels (Yes/No) found in test data")
        
    # Map labels
    test_df['label'] = test_df['contains_toxicity'].map({'No': 0, 'Yes': 1})
    return test_df

def test_model(model_path, test_df):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    # Make predictions
    with torch.no_grad():
        for text in test_df['audio_file_transcript']:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            predictions.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(test_df['label'], predictions)
    report = classification_report(test_df['label'], predictions)
    
    return accuracy, report

if __name__ == "__main__":
    model_path = "./toxicity_classifier"
    csv_path = "eng_hin.csv"
    
    try:
        df = pd.read_csv("eng_hin.csv")
        print(df['partition'].value_counts())
        print("\nToxicity labels in test partition:")
        print(df[df['partition'] == 'test']['contains_toxicity'].value_counts())
        
        test_df = load_test_data(csv_path)
        print(f"Found {len(test_df)} test samples")
        
        accuracy, report = test_model(model_path, test_df)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")