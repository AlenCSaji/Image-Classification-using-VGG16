# Yelp Image Classification using Pretrained VGG16 + Captions

---

## Objective

To classify Yelp business photos into categories — **Drink, Food, Inside, Outside** — using a **fine-tuned VGG16 CNN model** enhanced with business metadata (captions). The model combines visual and textual data for improved accuracy and explainability.

---

## Dataset

- **Source**: Yelp Open Dataset (Business Photos & Metadata)
- **Train/Val/Test Split**:
  - `/augmented_photos/train/`
  - `/split_photos/val/`
  - `/split_photos/test/`
- **Classes**: Drink, Food, Inside, Outside
- **Metadata**: Caption (text) from `train_augmented_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`

---

## Model Architecture

### Image Branch
- Pretrained `VGG16` (weights from ImageNet, include_top=False)
- Global Average Pooling
- Fully connected Dense layers
- Fine-tuned top layers

###  Caption Branch
- Text input tokenized and embedded
- Dense layers to learn caption representation

###  Merged Output
- Concatenated image and caption features
- Final Dense layer with softmax for classification

---

##  Training Details

- Image size: **128x128**
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- Class weights applied to handle imbalance
- EarlyStopping and validation monitoring
- Total train set: ~17K images

---

##  Interpretability

###  Grad-CAM (Image)
- Highlights discriminative regions in the input image
- Applied on the final VGG convolutional layer

###  LIME (Caption)
- Explains which words in the caption contributed to the classification
- Supports text + image interpretability

---

##  Metrics

| Model Variant              | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| VGG16 + Caption (Fine-tuned) | ~87%     | High      | High   | High     |
| VGG16 Only (Frozen)         | ~81%     | Moderate  | Moderate| Moderate |

---

## 🛠️ Tools & Stack

| Category         | Tools Used                        |
|------------------|-----------------------------------|
| Programming      | Python                            |
| DL Framework     | TensorFlow / Keras                |
| NLP              | NLTK, Keras Tokenizer, LIME       |
| Image Processing | OpenCV, PIL                       |
| Visualization    | Grad-CAM, Matplotlib              |
| Evaluation       | Confusion Matrix, Classification Report |
| Deployment Ready | Flask / Streamlit                 |

---
