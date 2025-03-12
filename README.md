# **Image Classification on CIFAR-10 using AlexNet, VGG16, and ResNet50**  

## 📌 **Overview**  
This repository implements three deep learning models—AlexNet, VGG16, and ResNet50—for image classification on the CIFAR-10 dataset using PyTorch. The models are trained and evaluated to compare their performance in terms of accuracy.

## 🗂 **Dataset**  
- **CIFAR-10**: Contains 60,000 images across 10 classes, with 6,000 images per class.  
- Preprocessing includes normalization and data augmentation.

## ⚙️ **Implementation Details**  
- **Framework**: PyTorch  
- **Loss Function**: Cross-Entropy Loss  
- **Optimizer**: Adam  
- **Data Augmentation**: Random Crop, Horizontal Flip  
- **Batch Size**: 128  

## 📊 **Results**  

| Model    | Accuracy (%) |
|----------|-------------|
| **AlexNet**  | 86.17%      |
| **VGG16**    | 90.90%      |
| **ResNet50** | 91.08%      |

## 📌 **Conclusion**  
- ResNet50 achieved the highest accuracy (91.08%), followed by VGG16 (90.90%) and AlexNet (86.17%).  
- Deeper architectures (VGG16, ResNet50) performed better than AlexNet due to better feature extraction.  
