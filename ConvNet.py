import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
import random
from PIL import Image
import json
from pathlib import Path
import time
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'image_size': 256,
    'batch_size': 32,
    'learning_rate': 0.001,
    'dropout_rate': 0.5,
    'num_epochs': 10,
    'num_classes': 15,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'weight_decay': 1e-4,
    'patience': 3,
    'min_lr': 1e-6
}

# Expected classes (actual folder names)
EXPECTED_CLASSES = [
    '광선각화증', '기저세포암', '멜라닌세포모반', '보웬병', '비립종', 
    '사마귀', '악성흑색종', '지루각화증', '편평세포암', '표피낭종', 
    '피부섬유종', '피지샘증식증', '혈관종', '화농 육아종', '흑색점'
]

class SkinDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, data_type='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type  # 'train', 'val', or 'test'
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        
        # Load data based on data_type
        self._load_data()
        
    def _load_data(self):
        if self.data_type == 'train':
            # Load only training data
            data_dir = os.path.join(self.root_dir, 'Training', '01.원천데이터')
            prefix = 'TS_'
        elif self.data_type == 'test':
            # Load validation data as test set
            data_dir = os.path.join(self.root_dir, 'Validation', '01.원천데이터')
            prefix = 'VS_'
        else:
            raise ValueError(f"Invalid data_type: {self.data_type}")
        
        # Get class names from directory
        class_dirs = [d for d in os.listdir(data_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(data_dir, d))]
        class_names = [d.replace(prefix, '') for d in class_dirs]
        
        # Sort class names to ensure consistent ordering
        class_names = sorted(class_names)
        
        # Verify we have exactly 15 classes
        if len(class_names) != 15:
            raise ValueError(f"Expected 15 classes, but found {len(class_names)}: {class_names}")
        
        # Check if all expected classes are present
        missing_classes = set(EXPECTED_CLASSES) - set(class_names)
        if missing_classes:
            print(f"Warning: Missing classes: {missing_classes}")
            print(f"Found classes: {class_names}")
            # Use found classes instead of raising error
            class_names = sorted(class_names)
        
        self.classes = sorted(class_names)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes in {self.data_type} data: {self.classes}")
        
        # Load images
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, f'{prefix}{class_name}')
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total {self.data_type} images loaded: {len(self.images)}")
        
    def set_transform(self, transform):
        """Set transform for this dataset"""
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ConvNet(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.5):
        super(ConvNet, self).__init__()
        
        self.features = nn.Sequential(
            # First block: Conv(3→64)→BN→ReLU→MaxPool
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block: [Conv(64→128)→BN→ReLU]×2→MaxPool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block: [Conv(128→256)→BN→ReLU]×2→MaxPool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # He initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        return False

def get_transforms(image_size=256):
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms
    val_transform = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def calculate_class_weights(labels):
    """Calculate class weights based on frequency"""
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / np.log1p(counts)
    return torch.FloatTensor(class_weights)

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device, classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and probabilities for metrics
            probs = torch.softmax(output, dim=1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    
    # Macro metrics
    macro_f1 = f1_score(all_targets, all_predictions, average='macro')
    macro_recall = recall_score(all_targets, all_predictions, average='macro')
    
    # ROC-AUC (macro)
    try:
        macro_auc = roc_auc_score(all_targets, all_probs, average='macro', multi_class='ovr')
    except:
        macro_auc = 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_recall': macro_recall,
        'macro_auc': macro_auc,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probs
    }

def plot_training_curves(history, save_path):
    """Plot training curves and save metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Macro F1
    axes[0, 2].plot(history['val_macro_f1'], label='Validation')
    axes[0, 2].set_title('Macro F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Macro F1')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Macro Recall
    axes[1, 0].plot(history['val_macro_recall'], label='Validation')
    axes[1, 0].set_title('Macro Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Macro Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Macro AUC
    axes[1, 1].plot(history['val_macro_auc'], label='Validation')
    axes[1, 1].set_title('Macro ROC-AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Macro AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning Rate
    axes[1, 2].plot(history['lr'], label='Learning Rate')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('LR')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_csv(history, save_path):
    """Save metrics to CSV with delta values"""
    df = pd.DataFrame(history)
    
    # Calculate deltas
    for col in ['val_macro_f1', 'val_macro_recall', 'val_macro_auc']:
        df[f'{col}_delta'] = df[col].diff()
    
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_prob, classes, save_path):
    """Plot ROC curves for each class"""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        auc = roc_auc_score(y_true == i, y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

class BootstrappedDataset(Dataset):
    """Dataset wrapper for bootstrapping - creates bootstrap samples with replacement"""
    def __init__(self, dataset, bootstrap_size=None, random_seed=None):
        self.dataset = dataset
        self.bootstrap_size = bootstrap_size if bootstrap_size else len(dataset)
        self.random_seed = random_seed
        
        # Create bootstrap indices
        if random_seed is not None:
            np.random.seed(random_seed)
        self.bootstrap_indices = np.random.choice(len(dataset), size=self.bootstrap_size, replace=True)
        
    def __len__(self):
        return self.bootstrap_size
    
    def __getitem__(self, idx):
        return self.dataset[self.bootstrap_indices[idx]]

def main():
    # Data paths
    data_root = "15.피부종양 이미지 합성데이터/3.개방데이터/1.데이터"
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SkinDiseaseDataset(data_root, transform=None, data_type='train')
    test_dataset = SkinDiseaseDataset(data_root, transform=None, data_type='test')
    
    # Split training data into train and validation
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)), 
        test_size=1-CONFIG['train_split'], 
        stratify=train_dataset.labels,
        random_state=42
    )
    
    print(f"Training data split: Train {len(train_idx)}, Val {len(val_idx)}")
    print(f"Test data: {len(test_dataset)}")
    
    # Get transforms
    train_transform, val_transform = get_transforms(CONFIG['image_size'])
    
    # Create train and validation datasets from training data
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    
    # Set transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform
    test_dataset.transform = val_transform
    
    # Create bootstrapped training dataset
    print("Creating bootstrapped training dataset...")
    bootstrapped_train = BootstrappedDataset(
        train_subset, 
        bootstrap_size=len(train_subset), 
        random_seed=42
    )
    
    # Calculate class weights from training data
    train_labels = [train_dataset.labels[i] for i in train_idx]
    class_weights = calculate_class_weights(train_labels)
    print(f"Class weights: {class_weights}")
    
    # Create samplers and dataloaders
    sampler = WeightedRandomSampler(
        weights=[class_weights[label] for label in train_labels],
        num_samples=len(train_labels),
        replacement=True
    )
    
    train_loader = DataLoader(bootstrapped_train, batch_size=CONFIG['batch_size'], 
                            sampler=sampler, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], 
                          shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0, pin_memory=False)
    
    # Initialize model
    model = ConvNet(num_classes=CONFIG['num_classes'], dropout_rate=CONFIG['dropout_rate'])
    model = model.to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        betas=(0.9, 0.999), 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=CONFIG['min_lr']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['patience'])
    
    # Gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_macro_f1': [], 'val_macro_recall': [], 'val_macro_auc': [],
        'lr': []
    }
    
    print("Starting training with bootstrapped data...")
    best_f1 = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Create new bootstrap sample for each epoch
        if epoch > 0:
            bootstrapped_train = BootstrappedDataset(
                train_subset, 
                bootstrap_size=len(train_subset), 
                random_seed=42 + epoch
            )
            train_loader = DataLoader(bootstrapped_train, batch_size=CONFIG['batch_size'], 
                                   sampler=sampler, num_workers=0, pin_memory=False)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, train_dataset.classes)
        
        # Update scheduler
        scheduler.step(val_metrics['macro_f1'])
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_macro_recall'].append(val_metrics['macro_recall'])
        history['val_macro_auc'].append(val_metrics['macro_auc'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Macro Recall: {val_metrics['macro_recall']:.4f}")
        print(f"Val Macro AUC: {val_metrics['macro_auc']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_best.pt'))
            print(f"New best model saved with Macro F1: {best_f1:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['macro_f1'], model):
            print("Early stopping triggered")
            break
    
    # Save training curves
    plot_training_curves(history, os.path.join(output_dir, 'training_curves.png'))
    save_metrics_csv(history, os.path.join(output_dir, 'metrics.csv'))
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint_best.pt')))
    
    # Test evaluation
    print("\nEvaluating on test set (original validation data)...")
    test_metrics = validate_epoch(model, test_loader, criterion, device, test_dataset.classes)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"Macro AUC: {test_metrics['macro_auc']:.4f}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Test Results (Original Validation Data):\n")
        f.write(f"Accuracy: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"Macro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Macro Recall: {test_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro AUC: {test_metrics['macro_auc']:.4f}\n")
    
    # Classification report
    report = classification_report(
        test_metrics['targets'], 
        test_metrics['predictions'], 
        target_names=test_dataset.classes,
        output_dict=True
    )
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics['targets'], 
        test_metrics['predictions'], 
        test_dataset.classes,
        os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        np.array(test_metrics['targets']), 
        np.array(test_metrics['probabilities']), 
        test_dataset.classes,
        os.path.join(output_dir, 'roc_curves.png')
    )
    
    # Check if performance meets requirements
    requirements_met = (
        test_metrics['macro_auc'] >= 0.995 and 
        test_metrics['macro_f1'] >= 0.995 and 
        test_metrics['macro_recall'] >= 0.995
    )
    
    print(f"\nPerformance Requirements Met: {requirements_met}")
    print(f"Target: Macro AUC ≥ 0.995, Macro F1 ≥ 0.995, Macro Recall ≥ 0.995")
    print(f"Actual: Macro AUC = {test_metrics['macro_auc']:.4f}, Macro F1 = {test_metrics['macro_f1']:.4f}, Macro Recall = {test_metrics['macro_recall']:.4f}")
    
    if not requirements_met:
        print("\nPerformance requirements not met. Consider hyperparameter tuning.")
    
    print(f"\nAll results saved to {output_dir}/")

if __name__ == "__main__":
    main()
