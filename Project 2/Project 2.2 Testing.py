# -*- coding: utf-8 -*-
"""
author: Miguel Lopez
501035749
AER850 Project 2
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#%% Step 5 - Model testing
model_path = 'saved_models/tuned_model_2024-11-18_00-33-30.h5'  # Your model file
model = load_model(model_path, compile=False)
print(f"Model loaded from: {model_path}")

#%% Step 5.1  Image Preprocessing 
def img_transformer(img_path):
    """
    Preprocesses the image to match the input format for the model.
    """
    img_width, img_height = 500, 500 
    img = image.load_img(img_path, target_size=(img_width, img_height))  
    x = image.img_to_array(img)  
    x = x / 255  
    x = np.expand_dims(x, axis=0)  
    return np.vstack([x])  

# Extract the true label from the folder name
def true_class_label(img_path):
   
    if 'crack' in img_path:
        return 'crack'
    elif 'paint-off' in img_path:
        return 'paint-off'
    elif 'missing-head' in img_path:
        return 'missing-head'
    else:
        return 'unknown'

# Define prediction label
def pred_class_label(class_probs):
    class_labels = ['crack', 'missing-head', 'paint-off']
    predicted_class_index = np.argmax(class_probs)
    return class_labels[predicted_class_index]

#%% Step 5.2 - Test Images
test_images = {
    'data/test/crack/test_crack.jpg': 'crack',
    'data/test/missing-head/test_missinghead.jpg': 'missing-head',
    'data/test/paint-off/test_paintoff.jpg': 'paint-off'
}

# Iterate over the test images
for image_path, true_label in test_images.items():
    print(f"Testing image: {image_path}")
    
    # Preprocess and predict the image
    class_probs = model.predict(img_transformer(image_path))
    class_probs *= 100  # Convert probabilities to percentages
    
    # Get the predicted label based on model output
    predicted_label = pred_class_label(class_probs)
    
    # Display the results
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Prediction Confidence: {class_probs[0][np.argmax(class_probs)]}%")
    
    # Read image using OpenCV
    img = cv2.imread(image_path)
    crack_str = str(np.around(class_probs[0][0], decimals=1)) + '%'
    paint_off_str = str(np.around(class_probs[0][2], decimals=1)) + '%'
    missing_head_str = str(np.around(class_probs[0][1], decimals=1)) + '%'
    textboxstr = f"Crack:{crack_str}\nMissing Head:{missing_head_str}\nPaint-off:{paint_off_str}"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=1.1)
    ax.set_title(f"\nTrue Label: {true_label}\nPredicted Label: {predicted_label}")
    ax.text(0.95, 0.01, textboxstr,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    

#%% Extra - Test All Images / comment out if not needed

test_directories = ['data/test/crack', 'data/test/paint-off', 'data/test/missing-head']

for test_dir in test_directories:
    print(f"\nTesting all images in {test_dir}...")
    
    for image_file in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_file)
        
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            print(f"Testing image: {image_path}")
            
           
            class_probs = model.predict(img_transformer(image_path))
            class_probs *= 100 
            true_label = true_class_label(image_path)
            predicted_label = pred_class_label(class_probs)

            print(f"True Label: {true_label}")
            print(f"Predicted Label: {predicted_label}")
            print(f"Prediction Confidence: {class_probs[0][np.argmax(class_probs)]}%")

            img = cv2.imread(image_path)
            crack_str = str(np.around(class_probs[0][0], decimals=1)) + '%'
            paint_off_str = str(np.around(class_probs[0][2], decimals=1)) + '%'
            missing_head_str = str(np.around(class_probs[0][1], decimals=1)) + '%'
            
            textboxstr = f"Crack:{crack_str}\nMissing Head:{missing_head_str}\nPaint-off:{paint_off_str}"

            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=1.1)

            ax.set_title(f"\nTrue Label: {true_label}\nPredicted Label: {predicted_label}")
            ax.text(0.95, 0.01, textboxstr,
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='green', fontsize=15)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
