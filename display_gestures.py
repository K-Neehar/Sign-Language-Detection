import cv2
import os
from glob import glob

def display_gestures():
    """Displays all stored gesture images"""
    gesture_folders = sorted(glob("gestures/*/"))
    
    if not gesture_folders:
        print("No gesture images found. Please create gestures first.")
        return

    for folder in gesture_folders:
        image_files = sorted(glob(os.path.join(folder, "*.jpg")))

        if not image_files:
            print(f"No images found in {folder}. Skipping...")
            continue
        
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is None:
                print(f"Could not read {image_file}. Skipping...")
                continue

            cv2.imshow("Gesture Display", img)
            key = cv2.waitKey(500)  # Show each image for 500ms

            # Press 'q' to quit early
            if key == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()

display_gestures()
