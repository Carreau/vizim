import json
import numpy as np

# Read the JSON file
with open('grid_numbers.json', 'r') as file:
    data = json.load(file)
print(data)



# Convert to numpy array and replace empty strings with None
grid_array = np.array([[None if cell == ' ' else cell for cell in row] for row in data])
print(grid_array)

# Get dimensions of current grid
rows, cols = grid_array.shape

# Create new padded grid with None values
padded_grid = np.full((rows + 2, cols + 2), None)

# Insert original grid into center of padded grid
padded_grid[1:-1, 1:-1] = grid_array

print("\nPadded grid:")
print(padded_grid)


# Create a copy of padded grid for drawing
drawing = np.full_like(padded_grid, None, dtype=object)

# Set all border values to 0
drawing[0, :] = 0  # Top row
drawing[-1, :] = 0  # Bottom row
drawing[:, 0] = 0  # Left column 
drawing[:, -1] = 0  # Right column



print("\nDrawing grid with 0 borders:")
# full cursor ascci charater 
def show(drawing):
    for row in drawing[1:-1, 1:-1]:
        print(''.join('█' if cell == 1 else ('.' if cell is None else '-') for cell in row), end='|\n')


def one(drawing, padded_grid):
    # Loop through all cells in the drawing grid
    for i in range(1, drawing.shape[0]-1):
        for j in range(1, drawing.shape[1]-1):
            # Get the current cell and 8 neighboring cells
            current = drawing[i,j]
            neighbors = [
                drawing[i-1, j-1], drawing[i-1, j], drawing[i-1, j+1],
                drawing[i, j-1],   current,         drawing[i, j+1],
                drawing[i+1, j-1], drawing[i+1, j], drawing[i+1, j+1]
            ]
            
            # Count zeros and ones including the cell itself
            zeros = sum(1 for n in neighbors if n == 0)
            ones = sum(1 for n in neighbors if n == 1)
            # Get the number from the padded grid that indicates how many cells should be 1
            target_ones = padded_grid[i,j]
            if target_ones is None:
                continue
            print(target_ones)
            
            if 9 - zeros == target_ones:
                print(f"   Cell ({i-1},{j-1}) has {zeros} zeros and {ones} ones (including itself), and target is {target_ones}, filling in the missing ones")
                # Get indices of all non-None neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if drawing[i+di, j+dj] is None:
                            drawing[i+di, j+dj] = 1
            if ones == target_ones:
                print(f"   Cell ({i-1},{j-1}) has {zeros} zeros and {ones} ones (including itself), and target is {target_ones}, filling in the missing zeroes")
                # Get indices of all non-None neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if drawing[i+di, j+dj] is None:
                            drawing[i+di, j+dj] = 0

            
            #print(f"Cell ({i-1},{j-1}) has {zeros} zeros and {ones} ones (including itself)")

print("\033[H\033[J")
show(drawing)
input()
for i in range(40):
    one(drawing, padded_grid)
    print("\033[H\033[J")
    show(drawing)
    input()
# clear screen
#one(drawing, padded_grid)
#show(drawing)

