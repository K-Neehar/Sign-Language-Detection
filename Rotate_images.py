import cv2
import os

def flip_images():
    gest_folder = "gestures"
    
    for g_id in os.listdir(gest_folder):
        folder_path = os.path.join(gest_folder, g_id)
        images = sorted(os.listdir(folder_path))  # Get sorted list of images
        
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            new_path = os.path.join(folder_path, f"{os.path.splitext(img_name)[0]}_flipped.jpg")

            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Skipping missing image: {img_path}")
                continue
            
            flipped_img = cv2.flip(img, 1)
            cv2.imwrite(new_path, flipped_img)
            print(f"Saved flipped image: {new_path}")

flip_images()
