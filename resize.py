# preprocess.py
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(data_dir, img_size=128):
    """
    Load images and convert them to numerical arrays
    """
    images = []
    labels = []
    
    # Load cat images (label = 0)
    cat_dir = os.path.join(data_dir, 'cats')
    for img_name in os.listdir(cat_dir)[:1000]:  # Use first 1000 images
        img_path = os.path.join(cat_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to fixed size
            img = cv2.resize(img, (img_size, img_size))
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(0)  # Cat
    
    # Load dog images (label = 1)
    dog_dir = os.path.join(data_dir, 'dogs')
    for img_name in os.listdir(dog_dir)[:1000]:  # Use first 1000 images
        img_path = os.path.join(dog_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(1)  # Dog
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize pixel values to 0-1
    images = images / 255.0
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    
    return images, labels

if __name__ == "__main__":
    X, y = load_and_preprocess_images('data/train')
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    print("Data preprocessed and saved!")