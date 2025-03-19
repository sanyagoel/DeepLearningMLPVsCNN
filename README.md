🌸 Flower Classification Using MLP vs CNN
A project comparing two neural network approaches—MLP and CNN—for classifying images of four types of flowers: Crocus, Pansy, Daisy, and Sunflower. Using a dataset of 234 RGB images, we aim to explore how manual feature extraction stacks up against automated feature learning, and what it means for performance, efficiency, and accuracy.

🚀 Problem Statement
Can traditional Multi-Layer Perceptrons (MLPs) using handcrafted features compete with Convolutional Neural Networks (CNNs) that automatically learn features from raw data?

We set out to answer this by:

Preprocessing and feeding binary masked images into an MLP.
Feeding raw, unmasked images directly into a CNN.
Evaluating both models on accuracy, computational efficiency, and feature representation.
📦 Dataset Description
The dataset contains 234 images across four flower classes:

🌻 Sunflowers
🌼 Daisies
🌸 Crocuses
🟣 Pansies
Structure:
images/ – Raw RGB images of flowers.
masks/ – Binary masks highlighting just the flower (used for MLP).
Each image filename indicates its class, and labels were extracted using regular expressions (with Python’s re module). This allowed clean and automated label generation for training.

🛠️ Preprocessing Overview
For MLP:
Applied binary masks to isolate flowers.
Extracted color histograms (Hue, Saturation, Brightness).
Normalized extracted features before feeding them into the MLP.
For CNN:
Used raw, unmasked images.
Normalized pixel values (0–255 scaled to 0–1) to stabilize training.
🧠 Model Implementation
1. MLP Approach:
Input: Manually extracted features (color histograms).
Architecture: Multi-layer perceptron with hidden layers.
Result:
Training Accuracy: 73%
Testing Accuracy: 83%
2. CNN Approach:
Input: Raw images (no manual feature extraction).
Architecture: Convolutional layers, followed by dense layers with dropout.
Result:
Training Accuracy: 99%
Testing Accuracy: 96%
Validation Accuracy: 92%
📊 Key Learnings & Insights
1️⃣ Manual Feature Extraction Wasn’t Enough
Color histograms fed into the MLP gave limited classification power. While better than random guessing, it couldn’t handle complex patterns, especially given the small dataset.

2️⃣ CNNs Shined on Raw Images
CNNs automatically learned meaningful features and vastly outperformed the MLP. No manual feature selection was needed. This made a big difference in real-world applicability.

3️⃣ Overfitting Was Real
The CNN’s 99% training accuracy raised red flags for overfitting. To address this:

Increased dropout in fully connected layers from 40% to 50%.
Reduced the number of neurons.
Helped reduce overfitting while maintaining high accuracy.
4️⃣ Normalization Changed Everything
Before normalization, training was unstable—accuracy fluctuated wildly (e.g., dropping from 92% to 68%).
After scaling pixels to 0–1, gradients stabilized and the model trained smoothly. This proved how crucial normalization is in deep learning.

📈 Performance Metrics
MLP: Slower training, lower accuracy, limited feature learning.
CNN: Fast convergence, high accuracy, strong feature representation.
Evaluation: Confusion matrices, accuracy scores, and training graphs were used to compare both models.
✅ Conclusion
CNNs are far superior for image classification tasks, especially with small datasets, due to their ability to learn hierarchical features.
MLPs, while simpler, struggle without strong manual feature engineering.
Data normalization and dropout regularization are essential for model stability and performance.
💻 Tech Stack
Python, TensorFlow/Keras
NumPy, Matplotlib
Regex for label extraction
Jupyter Notebook for experimentation
