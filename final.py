import cv2
import numpy as np
import tensorflow as tf
import os
import pickle
import sqlite3
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load trained model
model_path = "cnn_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError(f"❌ Model file '{model_path}' not found. Train the model first.")

# Load hand histogram for background removal
def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

# 🔹 NEW: Image Preprocessing - Enhance contrast & remove noise
def enhance_image(img):
    img = cv2.equalizeHist(img)  # Improves contrast
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduces noise
    return img

# Prepare image for CNN
def keras_process_image(img):
    img = cv2.resize(img, (50, 50))  # Resize to match model input
    img = enhance_image(img)  # 🔹 APPLY PREPROCESSING
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img = np.reshape(img, (1, 50, 50, 1))
    return img

# Predict gesture using CNN
def keras_predict(image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = np.argmax(pred_probab)

    # # 🔹 NEW: Ignore low-confidence predictions
    # if pred_probab[pred_class] < 0.6:  # Threshold (Change 0.6 to adjust sensitivity)
    #     return 0, -1  # Invalid prediction
    # print(pred_class,pred_probab[pred_class])
    return pred_probab[pred_class], pred_class

# Fetch gesture text from database
def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    result = cursor.fetchone()
    # print(result)
    return result[0] if result else "Unknown"

# Start gesture recognition using webcam
def recognize():
    hist = get_hand_hist()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("❌ Could not access webcam. Ensure it's connected.")

    x, y, w, h = 400, 100, 250, 250  # Green box position

    while True:
        ret, img = cam.read()
        if not ret:
            print("⚠️ Failed to capture image.")
            continue

        img = cv2.flip(img, 1)  # Mirror the feed
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

        # Apply filters
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 🔹 NEW: Remove extra noise using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        roi_thresh = cv2.morphologyEx(thresh[y:y+h, x:x+w], cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(roi_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        text = ""

        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            # 🔹 NEW: Minimum area check to avoid false positives
            if area > 5000:
                pred_probab, pred_class = keras_predict(roi_thresh)
                
                # 🔹 NEW: Ignore low-confidence predictions
                # print(pred_class,pred_probab)
                text = get_pred_text_from_db(pred_class)
                if text != "Unknown":
                    print(f"✅ Prediction: {text},Prob: {pred_probab:.2f}")

        # Draw Green Box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show Prediction Text Inside the Green Box
        if text:
            cv2.putText(img, text, (x + 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display Windows
        cv2.imshow("Gesture Recognition", img)
        cv2.imshow("Thresholded", roi_thresh)  # Only for hand area

        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

recognize()
