from PIL import Image
import os


def combine_images(input_directory, output_path):
    # Get the file paths of all cropped images
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    # Get the size of the cropped images, assuming all sub-images have the same size
    first_image = Image.open(os.path.join(input_directory, image_files[0]))
    image_width, image_height = first_image.size

    # Calculate the size of the combined image
    combined_width = image_width * 4  # 4 columns
    combined_height = image_height * 4  # 4 rows

    # Create a new image for combination
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Traverse the cropped images and paste them into the combined image
    for i in range(4):
        for j in range(4):
            # Open the cropped image
            subimage_path = os.path.join(input_directory, f"subimage_{i}_{j}.png")
            subimage = Image.open(subimage_path)

            # Calculate the paste position
            left = j * image_width
            top = i * image_height

            # Paste the sub-image into the combined image
            combined_image.paste(subimage, (left, top))

    # Save the combined image
    combined_image.save(output_path)


# Example usage
input_directory = "imgresult/output_images7"  # Folder containing the cropped images
output_image_path = "imgresult/combined_image.jpg"  # Path for saving the combined image
combine_images(input_directory, output_image_path)
```
