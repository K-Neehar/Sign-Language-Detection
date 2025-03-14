import cv2
import numpy as np
import pickle
import os
from glob import glob
from sklearn.utils import shuffle

def load_images_labels():
    images_labels = []
    label_dict = {}  # Dictionary to store label mappings
    label_counter = 0  # Start numbering labels

    for image in sorted(glob("gestures/*/*.jpg")):
        folder_name = os.path.basename(os.path.dirname(image))  # Get folder name
        
        if folder_name not in label_dict:
            label_dict[folder_name] = label_counter
            label_counter += 1
        
        label = label_dict[folder_name]  # Assign corresponding label
        img = cv2.imread(image, 0)
        images_labels.append((np.array(img, dtype=np.uint8), label))
    
    return shuffle(images_labels)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

images_labels = load_images_labels()
images, labels = zip(*images_labels)

# Splitting dataset
split1, split2 = int(5/6 * len(images)), int(11/12 * len(images))
datasets = {
    "train": (images[:split1], labels[:split1]),
    "test": (images[split1:split2], labels[split1:split2]),
    "val": (images[split2:], labels[split2:])
}

for name, (imgs, lbls) in datasets.items():
    save_pickle(imgs, f"{name}_images")
    save_pickle(lbls, f"{name}_labels")
    print(f"Saved {name}: {len(imgs)} samples")
