import numpy as np
import pickle
import cv2
import os
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

def get_image_size():
    """Dynamically get image size from dataset"""
    paths = glob('gestures/*/*.jpg')
    if not paths:
        raise FileNotFoundError("❌ No images found in 'gestures/' folder.")
    img = cv2.imread(paths[0], 0)
    return img.shape

def get_num_of_classes():
    """Count total number of gesture classes"""
    return len(glob('gestures/*'))

image_x, image_y = get_image_size()

def cnn_model():
    """Build and compile the CNN model"""
    num_of_classes = get_num_of_classes()
    
    model = Sequential([
        Conv2D(32, (3,3), input_shape=(image_x, image_y, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_of_classes, activation='softmax')  # ✅ Changed for multi-class classification
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])  # ✅ Changed loss function
    
    checkpoint = ModelCheckpoint("cnn_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint]

def train():
    """Train the CNN model using stored dataset"""
    # Load training data
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f))

    # Load validation data
    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f))

    # Reshape images for CNN input
    train_images = train_images.reshape((train_images.shape[0], image_x, image_y, 1))
    val_images = val_images.reshape((val_images.shape[0], image_x, image_y, 1))

    # ✅ Encode labels correctly
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)

    # ✅ Fix: Ensure correct number of classes
    num_classes = get_num_of_classes()  # Instead of len(np.unique(train_labels))
    
    # ✅ One-hot encode labels
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    val_labels = to_categorical(val_labels, num_classes=num_classes)

    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Training Labels Shape: {train_labels.shape}")
    print(f"✅ Validation Labels Shape: {val_labels.shape}")

    # Save label encoder for decoding predictions later
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Create and train the model
    model, callbacks_list = cnn_model()
    model.summary()
    model.fit(
        train_images, train_labels, 
        validation_data=(val_images, val_labels), 
        epochs=25, batch_size=64, 
        callbacks=callbacks_list
    )

    # Evaluate model
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"🎯 CNN Error: {100-scores[1]*100:.2f}%")

    # Save the final model
    model.save('cnn_model.h5')
    print("✅ Model successfully saved!")

# Run training
train()
