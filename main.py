import cv2
import numpy as np

# Read the image
image = cv2.imread('grid.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold to get binary image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
grid_contour = max(contours, key=cv2.contourArea)

# Get approximate polygon
epsilon = 0.02 * cv2.arcLength(grid_contour, True)
approx = cv2.approxPolyDP(grid_contour, epsilon, True)

# Ensure we have 4 corners
if len(approx) == 4:
    corners = np.float32(approx.reshape(4, 2))
    
    # Order points in clockwise order starting from top-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have smallest sum
    # Bottom-right will have largest sum
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    
    # Top-right will have smallest difference
    # Bottom-left will have largest difference
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    # Get width and height for the new image
    width = int(max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    ))
    height = int(max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    ))
    
    # Define destination points for perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # Calculate perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    # Convert warped image to grayscale and apply threshold
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped_thresh = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Sum the pixels along rows and columns
    row_sum = np.sum(warped_thresh, axis=1)
    col_sum = np.sum(warped_thresh, axis=0)
    
    # Find peaks in the sums (these represent grid lines)
    row_peaks = np.where(row_sum > 0.5 * np.max(row_sum))[0]
    col_peaks = np.where(col_sum > 0.5 * np.max(col_sum))[0]
    
    # Group nearby peaks to identify unique grid lines
    def group_peaks(peaks, min_distance=10):
        if len(peaks) == 0:
            return []
        groups = [[peaks[0]]]
        for peak in peaks[1:]:
            if peak - groups[-1][-1] <= min_distance:
                groups[-1].append(peak)
            else:
                groups.append([peak])
        return [int(np.mean(group)) for group in groups]
    
    row_lines = group_peaks(row_peaks)
    col_lines = group_peaks(col_peaks)
    
    # Number of cells is one less than number of lines
    num_rows = len(row_lines) - 1
    num_cols = len(col_lines) - 1
    
    print(f"Detected grid size: {num_rows} rows x {num_cols} columns")
    
    # Optional: Draw the detected lines on the warped image
    warped_with_lines = warped.copy()
    for row in row_lines:
        cv2.line(warped_with_lines, (0, row), (width, row), (0, 0, 255), 2)
    for col in col_lines:
        cv2.line(warped_with_lines, (col, 0), (col, height), (0, 0, 255), 2)
    
    # Show all images
    cv2.imshow('Original', image)
    cv2.imshow('Corrected Grid', warped)
    cv2.imshow('Detected Lines', warped_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not detect exactly 4 corners")
