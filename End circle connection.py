import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def process_image(image_path, show_endpoint_circles=True):
    # Read the binarised crack skeleton image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Extract endpoint coordinate information of the cracks
    endpoints = []

    # Traverse the image to find all endpoints
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if image[y, x] == 255:  # Foreground pixel
                # Count the number of foreground pixels in the 8-neighbourhood
                neighborhood = image[y-1:y+2, x-1:x+2]
                if np.sum(neighborhood == 255) == 2:  # Only two foreground pixels including itself
                    endpoints.append((x, y))

    # Circle radius
    R = 30

    # Create a colour image to display the result
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Generate a circle for each endpoint
    circles = [(endpoint, cv2.circle(np.zeros_like(image), (endpoint[0], endpoint[1]), R, 255, 1)) for endpoint in endpoints]

    # Find intersecting or tangent circles
    linked_endpoints = set()

    def is_intersecting(circle1, circle2):
        center1, _ = circle1
        center2, _ = circle2
        dist = distance.euclidean(center1, center2)
        return dist <= 2 * R

    for i, circle1 in enumerate(circles):
        for j, circle2 in enumerate(circles):
            if i >= j:
                continue
            if is_intersecting(circle1, circle2):
                linked_endpoints.add((i, j))

    # Find and draw the smallest circle containing all endpoints
    def smallest_enclosing_circle(points):
        (cx, cy), radius = cv2.minEnclosingCircle(np.array(points))
        return (int(cx), int(cy)), int(radius)

    for i, j in linked_endpoints:
        points = [endpoints[i], endpoints[j]]
        for k in range(len(circles)):
            if k != i and k != j and is_intersecting(circles[i], circles[k]) and is_intersecting(circles[j], circles[k]):
                points.append(endpoints[k])
        center, radius = smallest_enclosing_circle(points)
        cv2.circle(output_image, center, radius, (0, 255, 0), 1)  # Green circle
        for point in points:
            cv2.line(output_image, center, tuple(point), (0, 255, 0), 1)  # Green line

    # Draw endpoint circles and separate intersecting thin lines
    for i, circle1 in enumerate(circles):
        if i in [ep[0] for ep in linked_endpoints] or i in [ep[1] for ep in linked_endpoints]:
            continue
        for j, circle2 in enumerate(circles):
            if i >= j:
                continue
            if is_intersecting(circle1, circle2):
                cv2.line(output_image, tuple(endpoints[i]), tuple(endpoints[j]), (0, 255, 0), 1)  # Green line

    # Draw all endpoint circles
    if show_endpoint_circles:
        for endpoint in endpoints:
            cv2.circle(output_image, endpoint, R, (0, 255, 255), 1)  # Yellow circle
            cv2.circle(output_image, endpoint, 2, (0, 0, 255), -1)  # Red endpoint
    else:
        for endpoint in endpoints:
            cv2.circle(output_image, endpoint, 2, (0, 0, 255), -1)  # Red endpoint

    # Display the result
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
image_path = "F:/cyx/Deeplearning/MASK_RCNN_2.5.0-master/chonggou/skeleton_image.png"
process_image(image_path, show_endpoint_circles=False)  # Set to True to show endpoint circles, or False to hide them