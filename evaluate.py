import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import SceneClassifier
from torch.utils.data import DataLoader, TensorDataset

# Set Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')

def load_test_data(data_dir='scene_classifier/data'):
    data = np.load(f'{data_dir}/test_data.npy')
    labels = np.load(f'{data_dir}/test_labels.npy')
    raw_images = data.copy()
    data = torch.from_numpy(data).unsqueeze(1)
    labels = torch.from_numpy(labels)
    return raw_images, TensorDataset(data, labels)

def calculate_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
        
    return cm, precision, recall, f1

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['Open Space', 'Caution Zone', 'Narrow Corridor', 'Dynamic Obstacle']
    plots_dir = 'scene_classifier/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Load model
    model = SceneClassifier()
    model.load_state_dict(torch.load('scene_classifier/checkpoints/best_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    raw_images, test_set = load_test_data()
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics manually
    cm, precision, recall, f1 = calculate_metrics(all_labels, all_preds, len(classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 1. Confusion Matrix Heatmap (using Matplotlib)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/confusion_matrix.png', dpi=150)
    plt.close()

    # 2. Per-class Accuracy Bar Chart
    accuracies = cm_norm.diagonal()
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(classes, accuracies, color=colors)
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/class_accuracy.png', dpi=150)
    plt.close()

    # 3. 4x4 Sample Grid Visualization
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    indices = np.random.choice(len(all_labels), 16, replace=False)
    
    for i, idx in enumerate(indices):
        ax = axes[i // 4, i % 4]
        img = raw_images[idx]
        true_label = classes[all_labels[idx]]
        pred_label = classes[all_preds[idx]]
        
        ax.imshow(img, cmap='viridis')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
        ax.axis('off')
        
        # Border color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=color, fill=False, linewidth=4)
        ax.add_patch(rect)

    plt.suptitle('Sample Classifications (Green: Correct, Red: Incorrect)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{plots_dir}/sample_predictions.png', dpi=150)
    plt.close()

    # 4. Metrics Summary TXT
    with open('scene_classifier/metrics_summary.txt', 'w') as f:
        f.write("Scene Classification Metrics Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n")
        f.write("-" * 56 + "\n")
        for i, cls in enumerate(classes):
            f.write(f"{cls:<20} | {precision[i]:<10.4f} | {recall[i]:<10.4f} | {f1[i]:<10.4f}\n")
        
        avg_p = np.mean(precision)
        avg_r = np.mean(recall)
        avg_f = np.mean(f1)
        f.write("-" * 56 + "\n")
        f.write(f"{'Macro Average':<20} | {avg_p:<10.4f} | {avg_r:<10.4f} | {avg_f:<10.4f}\n")
    
    print("Evaluation complete. All results saved to scene_classifier/plots/ and metrics_summary.txt")

if __name__ == '__main__':
    evaluate()
