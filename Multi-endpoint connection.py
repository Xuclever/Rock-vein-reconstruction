import cv2
import numpy as np
from collections import defaultdict

def find_cracks(binary_image):
    # Use an edge detection method, such as Canny edge detection, to find cracks
    edges = cv2.Canny(binary_image, 50, 150)

    # Use the probabilistic Hough line transform to detect straight lines; parameters can be adjusted as needed
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    return lines

def compute_features(lines):
    features = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Calculate the crack length
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Calculate the crack angle
        features.append((length, angle))

    return features

def compute_similarity(features1, features2):
    # Calculate the similarity or matching degree between cracks
    similarity = 0
    # A similarity metric can be designed according to actual conditions, considering factors such as angle difference, distance, and length difference
    # Here, the Euclidean distance between crack length and angle is calculated simply
    length_diff = abs(features1[0] - features2[0])
    angle_diff = abs(features1[1] - features2[1])
    similarity = np.sqrt(length_diff**2 + angle_diff**2)

    return similarity

def match_cracks(features_list):
    matched_pairs = defaultdict(list)

    for i, features1 in enumerate(features_list):
        for j, features2 in enumerate(features_list):
            if i != j:
                similarity = compute_similarity(features1, features2)
                matched_pairs[i].append((j, similarity))

    # Sort the matched cracks and select the most similar crack as the matching target
    for key, value in matched_pairs.items():
        value.sort(key=lambda x: x[1])
        matched_pairs[key] = value[0][0]

    return matched_pairs

def grow_cracks(binary_image, lines, matched_pairs):
    # Select the starting crack
    start_crack_index = list(matched_pairs.keys())[0]

    # Use geometric calculation to grow other cracks along the direction of the starting crack
    spliced_image = np.zeros_like(binary_image)

    for i, line in enumerate(lines):
        if i == start_crack_index:
            # Draw the starting crack onto the spliced image
            x1, y1, x2, y2 = line[0]
            cv2.line(spliced_image, (x1, y1), (x2, y2), 255, 1)
            continue

        # Get the index of the matched crack
        matched_index = matched_pairs[i]

        # Draw the current crack and its matched crack onto the spliced image
        x1, y1, x2, y2 = line[0]
        cv2.line(spliced_image, (x1, y1), (x2, y2), 255, 1)

        # Get the endpoint coordinates of the matched crack
        matched_line = lines[matched_index]
        x3, y3, x4, y4 = matched_line[0]

        # Calculate the splicing position
        if abs(x2 - x1) > abs(y2 - y1):
            # If the crack is horizontal, grow it along the x-axis direction
            y3_new = y3 + (y2 - y1)
            y4_new = y4 + (y2 - y1)
            x3_new = x3 + (x2 - x1)
            x4_new = x4 + (x2 - x1)
        else:
            # Otherwise, grow it along the y-axis direction
            x3_new = x3 + (x2 - x1)
            x4_new = x4 + (x2 - x1)
            y3_new = y3 + (y2 - y1)
            y4_new = y4 + (y2 - y1)

        # Draw the spliced crack onto the spliced image
        cv2.line(spliced_image, (x3_new, y3_new), (x4_new, y4_new), 255, 1)

    return spliced_image

# Read the binarised image
binary_image = cv2.imread('D:/cyx2/MASK_RCNN_2.5.0-master/imgresult/output_images/2/medial_axis.png', cv2.IMREAD_GRAYSCALE)

# Find cracks in the binarised image and calculate crack features
lines = find_cracks(binary_image)
features_list = compute_features(lines)

# Calculate the similarity between cracks and perform crack matching
matched_pairs = match_cracks(features_list)

# Splice the cracks
spliced_image = grow_cracks(binary_image, lines, matched_pairs)

# Display the spliced image
cv2.imshow('Spliced Image', spliced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
