
#%%
import os
from pathlib import Path
import face_recognition


def load_images_from_folder(folder_path):
    """Load all images from a folder"""
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(Path(folder_path).glob(ext))
    return sorted(image_paths)
#%%
# Process all images in datasets folder
dataset_path = "./datasets"
image_paths = load_images_from_folder(dataset_path)

# Show number of images found
print(f"Found {len(image_paths)} images in {dataset_path}")

# Process each image pair
for i in range(0, len(image_paths) - 1, 2):
    known_path = image_paths[i]
    unknown_path = image_paths[i + 1]

    print(f"\nComparing {known_path.name} with {unknown_path.name}")

    known_image = face_recognition.load_image_file(str(known_path))
    unknown_image = face_recognition.load_image_file(str(unknown_path))

    # Get face encodings
    known_encoding = face_recognition.face_encodings(known_image)
    unknown_encoding = face_recognition.face_encodings(unknown_image)

    if len(known_encoding) == 0 or len(unknown_encoding) == 0:
        print("Could not find faces in one or both images")
        continue

    # Compare faces
    results = face_recognition.compare_faces([known_encoding[0]], unknown_encoding[0])
    distance = np.linalg.norm(known_encoding[0] - unknown_encoding[0])

    print(f"Match: {results[0]}, Distance: {distance:.2f}")
#%% md
# The code above:
# 1. Creates a function to load all images from the datasets folder
# 2. Processes images in pairs, comparing each consecutive pair
# 3. For each pair:
#    - Loads both images
#    - Gets face encodings 
#    - Compares faces using face_recognition
#    - Prints match result and distance metric
# 
#%%
import face_recognition

known_image = face_recognition.load_image_file("./datasets/img_003_part24-43_group_081.png")
unknown_image = face_recognition.load_image_file("./datasets/1000025970.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding, tolerance=0.16)
#%%
results
#%%
encA = face_recognition.face_encodings(known_image)[0]
encB = face_recognition.face_encodings(unknown_image)[0]
print(np.linalg.norm(encA - encB))