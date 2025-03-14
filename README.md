# Gesture Recognition Project

This project is designed for hand gesture recognition using a Convolutional Neural Network (CNN). The system captures hand gestures, processes the images, and trains a model to recognize different gestures.

## Prerequisites

Ensure you have Python installed along with the required dependencies. You can install them using:

```bash
pip install -r requirements.txt
```

## Steps to Run the Project

Execute the following commands in order:

```bash
python set_hand_histogram.py   # Step 1: Create a hand histogram for gesture recognition
python create_gestures.py      # Step 2: Capture and create gesture images
python Rotate_images.py        # Step 3: Rotate images for data augmentation
python load_images.py          # Step 4: Load images into the dataset
python display_gestures.py     # Step 5: Display gestures for verification
python cnn_model_train.py      # Step 6: Train the CNN model
python final.py                # Step 7: Run the final model for gesture recognition
```

## Project Overview

1. **Hand Histogram Creation:** Creates a color histogram to detect hand regions.
2. **Gesture Image Capture:** Captures images for different hand gestures.
3. **Image Augmentation:** Rotates images to increase dataset variability.
4. **Dataset Preparation:** Loads and organizes images for training.
5. **Gesture Display:** Verifies gesture images before training.
6. **CNN Model Training:** Trains a convolutional neural network for gesture classification.
7. **Final Execution:** Runs the trained model to recognize hand gestures in real-time.

## Notes

- Ensure your webcam is connected (if required for gesture capture).
- Modify configurations if needed in the respective script files.
- The trained model is stored in the project directory after training.

