from PIL import Image
import os

def crop_and_save_images(input_folder, output_folder):
    # Ensure that the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Process only image files with specific extensions
            input_path = os.path.join(input_folder, filename)

            # Open the image
            original_image = Image.open(input_path)

            # Get the width and height of the original image
            original_width, original_height = original_image.size

            # Calculate the cropping width and height
            crop_width = original_width // 6
            crop_height = original_height // 4

            # Create a folder to save the cropped images
            output_image_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(output_image_folder, exist_ok=True)

            # Traverse the rows and columns of the cropping regions
            for i in range(6):
                for j in range(4):
                    left = i * crop_width
                    top = j * crop_height
                    right = left + crop_width
                    bottom = top + crop_height

                    # Crop the image
                    cropped_image = original_image.crop((left, top, right, bottom))

                    # Build the save path
                    output_path = os.path.join(output_image_folder, f"crop_{i}_{j}.png")

                    # Save the cropped image with the highest quality
                    cropped_image.save(output_path, quality=95)

# Example usage
input_folder_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/output/preprocess images"
output_folder_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/output/corpping-img"
crop_and_save_images(input_folder_path, output_folder_path)
```
