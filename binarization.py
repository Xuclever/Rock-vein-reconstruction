import os
import cv2

# Input image folder and output folder
input_folder = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/imgresult/output_images"
output_folder = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/1111/yuce"

# Create the output folder
os.makedirs(output_folder, exist_ok=True)

# Get all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".jpg")]

# Process each image in a loop
for image_file in image_files:
    # Build the full path of the input image
    image_path = os.path.join(input_folder, image_file)

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Build the full path of the output image
    output_path = os.path.join(output_folder, f"binary_{image_file}")

    # Save the binarised image
    cv2.imwrite(output_path, binary_image)

    print(f"Binary image saved to: {output_path}")