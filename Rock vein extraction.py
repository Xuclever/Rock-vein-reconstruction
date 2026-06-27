from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage import io, measure
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import os

def find_endpoints(skeleton):
    """Find endpoints of the skeleton."""
    # Define a kernel to calculate the number of neighbours for each pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

    # Endpoints are pixels with only one neighbour
    endpoints = (skeleton & (neighbor_count == 1))
    return endpoints

def calculate_angle(p1, p2):
    """Calculate angle between the line p1p2 and the horizontal axis."""
    delta_y = p2[0] - p1[0]
    delta_x = p2[1] - p1[1]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi
    return angle

# Load the image and convert it to grayscale
path = r"F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/chonggou/output_image_CG.jpg"
image = io.imread(path, as_gray=True)

# Apply Otsu's thresholding method for binarisation
thresh = threshold_otsu(image)
binary = image > thresh

# Extract the skeleton
skeleton = skeletonize(binary)

# Label connected regions in the image
label_image = measure.label(skeleton)

# Calculate the skeleton length of each connected region and find its endpoints
region_props = measure.regionprops(label_image)
skeleton_lengths = []
endpoints_angles = []
endpoints_positions = []  # Store endpoint positions

for region in region_props:
    skeleton_lengths.append(region.perimeter)

    # Find endpoints
    endpoints_image = find_endpoints(region.image)
    endpoints_coords = np.argwhere(endpoints_image)

    # Convert endpoint coordinates to global image coordinates
    endpoints_coords_global = endpoints_coords + region.bbox[:2]
    endpoints_positions.append(endpoints_coords_global)

    # Calculate the angle between the endpoint connection line and the horizontal axis
    if len(endpoints_coords) == 2:
        p1, p2 = endpoints_coords
        angle = calculate_angle(p1, p2)
        endpoints_angles.append(angle)
    else:
        endpoints_angles.append(None)  # Angle cannot be calculated

# Print the skeleton length and endpoint angle of each crack
for i, (length, angle) in enumerate(zip(skeleton_lengths, endpoints_angles)):
    if angle is not None:
        print(f"裂隙 {i + 1} 的骨架长度: {length}, 端点连线与水平x轴的夹角: {angle:.2f} 度")
    else:
        print(f"裂隙 {i + 1} 的骨架长度: {length}, 端点连线与水平x轴的夹角: 无法计算")

# Display the original image and the skeleton image, and annotate crack positions
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('原始图像')
axes[1].imshow(skeleton, cmap='gray')
axes[1].set_title('骨架图像')

# Annotate crack positions and numbers on the original image
for i, coords in enumerate(endpoints_positions):
    if len(coords) > 0:
        y, x = coords[0]  # Use the first endpoint as the annotation position
        axes[0].text(x, y, str(i + 1), color='red', fontsize=12)

# Save the skeletonised image to drive F
output_path = r"F:/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

skeleton_image_path = os.path.join(output_path, "skeleton_image.png")
io.imsave(skeleton_image_path, skeleton.astype(np.uint8) * 255)
print(f"骨架化图像已保存到: {skeleton_image_path}")

plt.show()