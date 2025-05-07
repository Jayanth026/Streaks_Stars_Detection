# Import necessary libraries
import os  # For file and directory operations
import cv2  # OpenCV for image processing
import numpy as np  # Numerical operations
import pandas as pd  # Data handling with DataFrames
from skimage import io  # Image I/O operations
from skimage.measure import label, regionprops  # Image region analysis
from skimage.morphology import remove_small_objects  # Morphological operations

# Define input and output paths
input_folder= r"C:\Users\jayan\Downloads\Datasets_Assessment\Datasets\Raw_Images"
output_csv_folder= os.path.join(input_folder,"Centroid_CSVs")  # Folder for centroid CSVs
output_img_folder= os.path.join(input_folder,"Annotated_Images")  # Folder for annotated images

# Create output directories if they don't exist
os.makedirs(output_csv_folder, exist_ok=True)
os.makedirs(output_img_folder, exist_ok=True)

# === PROCESS EACH IMAGE ===
# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Only process TIFF files
    if not filename.lower().endswith((".tif",".tiff")):
        continue

    file_path = os.path.join(input_folder, filename)
    print(f"\n🔍 Processing: {filename}")

    # Load image as 16-bit grayscale
    image = io.imread(file_path).astype(np.uint16)
    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize to 8-bit for OpenCV processing
    image_norm= cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Apply Gaussian blur to reduce noise
    blurred= cv2.GaussianBlur(image_norm, (3, 3), 0)
    # Threshold to create binary image
    _, binary= cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    # Remove small objects from binary image
    cleaned= remove_small_objects(binary.astype(bool), min_size=5)
    # Label connected components
    labels= label(cleaned)

    # Initialize lists to store star and streak coordinates
    stars,streaks= [],[]
    # Create output image for visualization (3-channel color)
    output_img= np.zeros_like(cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR))

    # Analyze each region in the labeled image
    for region in regionprops(labels):
        # Skip regions that are too small or too large
        if region.area< 5 or region.area >10000:
            continue

        # Get centroid coordinates and eccentricity
        y,x= region.centroid
        ecc= region.eccentricity

        # Classify as star (round) or streak (elongated)
        if ecc <0.85:  # Stars have lower eccentricity
            stars.append((x,y))
            # Draw white dot for stars
            cv2.circle(output_img,(int(x), int(y)),2,(255,255,255),-1)
        else:  # Streaks have higher eccentricity
            streaks.append((x,y))
            # Draw line showing streak orientation
            orientation = region.orientation
            length = region.major_axis_length * 2
            x0,y0 = x,y
            x1= x0 + np.cos(orientation) * length / 2
            y1= y0 - np.sin(orientation) * length / 2
            x2= x0 - np.cos(orientation) * length / 2
            y2= y0 + np.sin(orientation) * length / 2
            h,w= output_img.shape[:2]
            pt1= (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            pt2= (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(output_img, pt1, pt2, (255, 255, 255), 1)

    print(f"✅ Stars: {len(stars)} | Streaks: {len(streaks)}")

    # Save centroid data to CSV
    df = pd.DataFrame(stars + streaks, columns=["x", "y"])
    df["type"]= ["star"] * len(stars) + ["streak"] * len(streaks)
    csv_path= os.path.join(output_csv_folder,filename.replace(".tiff","_centroids.csv").replace(".tif","_centroids.csv"))
    df.to_csv(csv_path, index=False)

    # Save annotated image
    image_name= os.path.splitext(filename)[0] + "_annotated.png"
    img_path= os.path.join(output_img_folder,image_name)
    cv2.imwrite(img_path,output_img)

print("\n✅ Done. All images processed with CSV and visual outputs.")

# Import machine learning related libraries
from sklearn.model_selection import train_test_split  # For splitting dataset
from tensorflow.keras.models import Sequential  # Keras model class
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization  # NN layers
from tensorflow.keras.optimizers import Adam  # Optimizer
import time  # For timing operations

# Initialize lists to store image patches and their labels
patches,labels= [], []

# === Load patches from centroid CSVs ===
# Process each image file to extract patches around detected objects
for file in os.listdir(input_folder):
    if file.endswith(".tif") or file.endswith(".tiff"):
        img_path= os.path.join(input_folder, file)
        base_name= os.path.splitext(file)[0]
        csv_path= os.path.join(output_csv_folder, base_name + "_centroids.csv")
        if not os.path.exists(csv_path):
            continue

        # Load and preprocess image
        image= io.imread(img_path).astype(np.uint16)
        if image.ndim == 3:
            image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Read centroid data from CSV
        df= pd.read_csv(csv_path)
        for _, row in df.iterrows():
            x, y= int(row['x']), int(row['y'])
            label= 0 if row['type'] == 'star' else 1  # 0 for stars, 1 for streaks

            # Extract 32x32 patch around each centroid
            half= 16
            if y - half >= 0 and y + half < image.shape[0] and x - half >= 0 and x + half < image.shape[1]:
                patch = image[y - half:y + half, x - half:x + half]
                patches.append(patch)
                labels.append(label)

# === Prepare dataset ===
# Convert to numpy arrays and normalize pixel values
X= np.array(patches).reshape(-1, 32, 32, 1).astype("float32") / 255.0
y= np.array(labels)
# Split into training and testing sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Set up data augmentation to improve model generalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen= ImageDataGenerator(rotation_range=20, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

model= Sequential([
    # First convolutional block
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    BatchNormalization(),  # Normalize activations
    MaxPooling2D((2, 2)),  # Downsampling

    # Second convolutional block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Classifier head
    Flatten(),  # Convert 3D features to 1D
    Dropout(0.4),  # Regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the model with augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32),validation_data=(X_test, y_test),epochs=15)

# Time the training process
start= time.time()
history= model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
train_time= time.time() - start

# Evaluate model performance
train_loss,train_acc= model.evaluate(X_train, y_train, verbose=0)
test_loss,test_acc= model.evaluate(X_test, y_test, verbose=0)

# Print results
print(f"\n✅ Training Accuracy: {train_acc:.4f}")
print(f"✅ Testing Accuracy: {test_acc:.4f}")
print(f"⏱️ Training Time: {train_time:.2f} seconds")

# Save the trained model
model.save("star_streak_classifier.h5")