import json
import numpy as np



def prepare(data):
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


    # Create a copy of padded grid for drawing
    drawing = np.full_like(padded_grid, None, dtype=object)

    # Set all border values to 0
    drawing[0, :] = 0  # Top row
    drawing[-1, :] = 0  # Bottom row
    drawing[:, 0] = 0  # Left column 
    drawing[:, -1] = 0  # Right column

    return padded_grid, drawing



# full cursor ascci charater 
def show(drawing):
    for row in drawing[1:-1, 1:-1]:
        print(''.join('█' if cell == 1 else ('.' if cell is None else '-') for cell in row), end='|\n')


def loop(drawing, padded_grid):
    for i in range(40):
        for step in one(drawing, padded_grid):
            yield step

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
            if ones + zeros == 9:
                continue
            
            if 9 - zeros == target_ones:

                yield i,j
                # Get indices of all non-None neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if drawing[i+di, j+dj] is None:
                            drawing[i+di, j+dj] = 1
                            yield
            if ones == target_ones:
                yield i,j
                # Get indices of all non-None neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if drawing[i+di, j+dj] is None:
                            drawing[i+di, j+dj] = 0
                            yield


if __name__ == "__main__":
# Read the JSON file
    with open('grid_numbers.json', 'r') as file:
        data = json.load(file)
    padded_grid, drawing = prepare(data)
    print("\033[H\033[J")
    show(drawing)
    input()
    for i in range(40):
        for step in one(drawing, padded_grid):
            print("\033[H\033[J")
            show(drawing)
            import time
            time.sleep(0.02)
            #input()
    # clear screen
    #one(drawing, padded_grid)
    #show(drawing)

