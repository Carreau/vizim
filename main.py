import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import itertools

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
if len(approx) != 4:
    print("Could not detect exactly 4 corners")
    sys.exit(1)

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

# Plot row and column sums for debugging
plt.figure(figsize=(12, 4))

THRESHOLD = 0.3

plt.subplot(121)
plt.plot(row_sum)
plt.title('Row Sums')
plt.xlabel('Row Index')
plt.ylabel('Sum of Pixels')
plt.axhline(y=THRESHOLD * np.max(row_sum), color='r', linestyle='--', label='Threshold')
plt.legend()

plt.subplot(122)
plt.plot(col_sum)
plt.title('Column Sums')
plt.xlabel('Column Index')
plt.ylabel('Sum of Pixels')
plt.axhline(y=THRESHOLD * np.max(col_sum), color='r', linestyle='--', label='Threshold')
plt.legend()

plt.tight_layout()
plt.show()

# Also try to detect grid size using FFT
def detect_grid_size_fft(image):
    # Convert to grayscale if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    # Get the 1D FFT for rows and columns
    row_fft = np.sum(magnitude_spectrum, axis=1)
    col_fft = np.sum(magnitude_spectrum, axis=0)
    
    # Find peaks in FFT
    from scipy.signal import find_peaks
    row_peaks, _ = find_peaks(row_fft, height=np.mean(row_fft))
    col_peaks, _ = find_peaks(col_fft, height=np.mean(col_fft))
    
    # The distance between peaks in frequency domain corresponds to grid spacing
    if len(row_peaks) > 1:
        row_spacing = np.median(np.diff(row_peaks))
        num_rows_fft = int(gray.shape[0] / row_spacing)
    else:
        num_rows_fft = 0
        
    if len(col_peaks) > 1:
        col_spacing = np.median(np.diff(col_peaks))
        num_cols_fft = int(gray.shape[1] / col_spacing)
    else:
        num_cols_fft = 0
    
    # Plot FFT results
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(row_fft)
    plt.title('Row FFT')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    
    plt.subplot(122)
    plt.plot(col_fft)
    plt.title('Column FFT')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()
    
    return num_rows_fft, num_cols_fft

# Get grid size estimation from FFT
num_rows_fft, num_cols_fft = detect_grid_size_fft(warped_thresh)
print(f"FFT estimated grid size: {num_rows_fft} rows x {num_cols_fft} columns")


# Find peaks in the sums (these represent grid lines)
row_peaks = np.where(row_sum > THRESHOLD * np.max(row_sum))[0]
col_peaks = np.where(col_sum > THRESHOLD * np.max(col_sum))[0]

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


# Now we find extract each cell and use OCR to extract the number in the cell if there is one
# Initialize a 2D array to store the extracted numbers
grid_numbers = [[None for _ in range(num_cols)] for _ in range(num_rows)]

# Iterate through each cell in the grid
from tqdm import tqdm
for idx, (i, j) in enumerate(tqdm(itertools.product(range(num_rows), range(num_cols)), total=num_rows*num_cols)):
        # Get coordinates for current cell
        y_start = row_lines[i]
        y_end = row_lines[i + 1]
        x_start = col_lines[j]
        x_end = col_lines[j + 1]
        
        # Extract the cell image
        cell = warped[y_start:y_end, x_start:x_end]
        
        # Add padding around the cell to improve OCR
        padding = 5
        cell_padded = cv2.copyMakeBorder(
            cell,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        
        # Convert to grayscale if not already
        if len(cell_padded.shape) == 3:
            cell_padded = cv2.cvtColor(cell_padded, cv2.COLOR_BGR2GRAY)
            
        # Threshold to get black text on white background
        _, cell_thresh = cv2.threshold(cell_padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to extract text
        try:
            import pytesseract
            # Show the thresholded cell for debugging
            cv2.imshow(f'Cell {i},{j}', cell_thresh)
            cv2.waitKey(1)  # Brief delay to allow window updates
            text = pytesseract.image_to_string(cell_thresh, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            # Clean the extracted text
            text = text.strip()
            if text:
                try:
                    number = int(text)
                    grid_numbers[i][j] = number
                except ValueError:
                    # If conversion to int fails, ignore this cell
                    pass
        except ImportError:
            print("Pytesseract not installed. Please install it to enable OCR functionality.")
            break

# Create a copy of the original image to draw on
grid_overlay = warped.copy()

# Draw the grid lines
for x in col_lines:
    cv2.line(grid_overlay, (x, 0), (x, warped.shape[0]), (0, 255, 0), 2)
for y in row_lines:
    cv2.line(grid_overlay, (0, y), (warped.shape[1], y), (0, 255, 0), 2)

# Draw detected numbers
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
font_thickness = 2

for i in range(num_rows):
    for j in range(num_cols):
        if grid_numbers[i][j] is not None:
            # Calculate center position of each cell
            cell_center_x = (col_lines[j] + col_lines[j + 1]) // 2
            cell_center_y = (row_lines[i] + row_lines[i + 1]) // 2
            
            # Convert number to string
            number_str = str(grid_numbers[i][j])
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(number_str, font, font_scale, font_thickness)
            
            # Calculate text position to center it in the cell
            text_x = cell_center_x - text_width // 2
            text_y = cell_center_y + text_height // 2
            
            # Draw the number
            cv2.putText(grid_overlay, number_str, (text_x, text_y), 
                       font, font_scale, (0, 0, 255), font_thickness)

# Display the overlay
cv2.imshow('Detected Grid and Numbers', grid_overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Print the extracted grid

print("\nExtracted Grid:")
for row in grid_numbers:
    print(row)
