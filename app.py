import tensorflow as tf
import numpy as np
from PIL import Image

# Define class labels
classes = {
    0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'basal cell carcinoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    3: ('df', 'dermatofibroma'),
    4: ('nv', 'melanocytic nevi'),
    5: ('vasc', 'pyogenic granulomas and hemorrhage'),
    6: ('mel', 'melanoma'),
}

# Load the saved model
model = tf.keras.models.load_model('best_model1.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize the image
    image = image.reshape(1, 28, 28, 3)  # Add batch dimension
    return image

# Predict function
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    max_prob = max(predictions[0])
    class_ind = np.argmax(predictions[0])
    predicted_class = classes[class_ind][0]
    description = classes[class_ind][1]
    return predicted_class, description, max_prob

# Test with an input image
if __name__ == "__main__":
    image_path = input("Enter the path of the image: ")
    try:
        predicted_class, description, probability = predict_image(image_path)
        print(f"Predicted Class: {predicted_class}")
        print(f"Description: {description}")
        print(f"Confidence: {probability * 100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")