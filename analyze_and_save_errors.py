"""
Script pour analyser toutes les images et sauvegarder les erreurs
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
import pandas as pd
from train_model import AlienCNN

def analyze_all_predictions():
    """Analyze all images and save errors to JSON"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    with open('models/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    model = AlienCNN(num_classes=metadata['num_classes'])
    state_dict = torch.load('models/alien_classifier_best.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load CSV
    csv_path = Path('images/class/classification.csv')
    df = pd.read_csv(csv_path)
    
    images_dir = Path('images/aliens')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {
        'correct': [],
        'errors': [],
        'stats': {}
    }
    
    print("Analyzing all images...")
    
    for idx, row in df.iterrows():
        img_num = str(row['Image']).zfill(3)
        true_label = row['Label']
        
        # Find image file
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            path = images_dir / f"{img_num}{ext}"
            if path.exists():
                img_path = str(path)
                break
        
        if not img_path:
            continue
        
        try:
            # Load and predict
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs, _ = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                pred_class = torch.argmax(probabilities).item()
                confidence = probabilities[pred_class].item()
                predicted = metadata['classes'][pred_class]
            
            # All probabilities
            all_probs = {metadata['classes'][i]: probabilities[i].item() 
                        for i in range(len(metadata['classes']))}
            
            entry = {
                'image_number': int(row['Image']),
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': predicted,
                'confidence': float(confidence),
                'all_probabilities': all_probs,
                'is_correct': predicted == true_label
            }
            
            if predicted == true_label:
                results['correct'].append(entry)
            else:
                results['errors'].append(entry)
            
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(df)} images...")
        
        except Exception as e:
            print(f"Error processing image {img_num}: {e}")
            continue
    
    # Calculate stats
    total = len(results['correct']) + len(results['errors'])
    accuracy = len(results['correct']) / total if total > 0 else 0
    
    # Confusion matrix
    confusion = {}
    for error in results['errors']:
        key = f"{error['true_label']} → {error['predicted_label']}"
        confusion[key] = confusion.get(key, 0) + 1
    
    results['stats'] = {
        'total_images': total,
        'correct_predictions': len(results['correct']),
        'errors': len(results['errors']),
        'accuracy': accuracy,
        'error_rate': 1 - accuracy,
        'confusion_matrix': confusion
    }
    
    # Save to JSON
    output_path = Path('models/prediction_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"Total images: {total}")
    print(f"Correct: {len(results['correct'])} ({accuracy*100:.1f}%)")
    print(f"Errors: {len(results['errors'])} ({(1-accuracy)*100:.1f}%)")
    print(f"\nResults saved to: {output_path}")
    print(f"\nMost common errors:")
    for error_type, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {error_type}: {count}")

if __name__ == '__main__':
    analyze_all_predictions()
