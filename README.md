Skin cancer is a major public health issue worldwide, and its effective treatment depends on its early detection. Using machine learning (ML) and artificial intelligence (AI) advances, researchers have investigated creating automated systems to help dermatologists correctly diagnose skin lesions. The most common cause for skin cancer detection is exposure to ultraviolet radiation which damages the DNA of skin cells resulting in uncontrollable cell growth which leads to cancer tumors which ultimately has a fatal conclusion. Therefore, skin cancer detection has become extremely crucial. A deep learning algorithm, Convolution Neural Network (CNN), is used to create the detection code as it specializes in processing and classifying visual data, making it ideal for tasks like image recognition.
1. Data Collection and Preprocessing: 
Collection: The Skin Cancer MNIST: HAM10000 Dataset. A total of 10015 dermatoscopic images of skin lesions are included in the HAM10000 dataset
Data Preprocessing: Prepare the data for the CNN:
•	Resizing: Ensure all images are the same size
•	Normalization: Scale pixel values to a range of 0-1.
•	Augmentation: Apply transformations (e.g., rotation, flipping) to increase the diversity of the training set and prevent overfitting.

2. Definition of Model Architecture 
•	Input Layer: Takes in the image data
•	Convolutional Layers: Apply filters to the input image to detect features like edges, textures, etc.
•	Activation Function: Apply non-linear functions like ReLU to introduce non-linearity.
•	Pooling Layers: Reduce the spatial dimensions of the feature maps, typically using max pooling.
•	Fully Connected Layers: Flatten the output from the convolutional layers and connect to fully connected (dense) layers.
•	Output Layer: Typically uses a softmax activation for classification tasks, providing probabilities for each class.
3. Model Training and Compilation: 
Compilation: Configure the model for training by specifying:
•	Optimizer: Algorithm to update model weights (e.g., Adam)
•	Metrics: Metrics to monitor during training (e.g., accuracy).
Training: Feed the training data into the model in batches, allowing it to learn:
•	Epochs: Complete passes through the entire training dataset.
•	Batch Size: Number of samples processed before the model is updated.
4. Assessment
Evaluation: Assess the model’s performance on a validation dataset that was not used during training:
•	Accuracy: Proportion of correctly classified images.
•	Loss: Measure of how well the model’s predictions match the true labels.
•	Confusion Matrix: Detailed breakdown of prediction results, showing true vs. predicted labels.
5. Optimization of the Model
•	Hyperparameter Tuning: Adjust parameters like learning rate, batch size, number of layers, etc., to improve performance.
•	Regularization: Techniques like dropout, L2 regularization to prevent overfitting.
•	Fine-tuning: Make small adjustments to a pre-trained model on a new, related task.

