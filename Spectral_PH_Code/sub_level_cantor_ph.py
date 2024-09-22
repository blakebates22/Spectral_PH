import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd

# Function to generate Cantor set points up to a given level
def cantor_set(level, start=0, end=1):
    if level == 0:
        return [start, end]
    else:
        third = (end - start) / 3
        return (cantor_set(level - 1, start, start + third) +
                cantor_set(level - 1, end - third, end))

# Function to compute the distance function for points in the Cantor set
def distance_function(x, cantor_points):
    return min(abs(x - point) for point in cantor_points)

# Generate Cantor set for 6 levels
level = 6
cantor_points = cantor_set(level)

# Create an array of x values
x_values = np.linspace(0, 1, 1000)
# Calculate the distance function for each x value
distance_values = np.array([distance_function(x, cantor_points) for x in x_values])

# Compute sublevel set persistence using GUDHI
def compute_persistence(distance_values):
    # Create a SimplexTree for GUDHI
    st = gd.SimplexTree()

    # Insert vertices with the distance values (for 0-simplices)
    for i, dist in enumerate(distance_values):
        st.insert([i], filtration=dist)

    # Insert edges between consecutive points (1-simplices)
    for i in range(len(distance_values) - 1):
        st.insert([i, i + 1], filtration=max(distance_values[i], distance_values[i + 1]))

    # Compute persistence (0-dimensional persistence)
    st.compute_persistence()

    # Extract 0-dimensional persistence pairs (connected components)
    return st.persistence_intervals_in_dimension(0)

# Step 2: Compute persistence for the distance function
persistence_pairs = compute_persistence(distance_values)

# Step 3: Track the number of connected components (minima) as a function of sublevel set cutoff
def count_components(persistence_pairs, cutoff):
    return sum(1 for birth, death in persistence_pairs if birth <= cutoff < death)

# Step through different sublevel set cutoff values
cutoff_values = np.linspace(0, np.max(distance_values), 100)  # 100 cutoff values
component_counts = [count_components(persistence_pairs, cutoff) for cutoff in cutoff_values]

# Plot the number of components (minima) as a function of the cutoff value
plt.figure(figsize=(10, 6))
plt.plot(cutoff_values, component_counts, label='Number of Connected Components')
plt.xlabel('Sublevel Set Cutoff')
plt.ylabel('Number of Connected Components')
plt.title('Number of Connected Components vs. Sublevel Set Cutoff for Cantor Set Distance Function')
plt.grid(True)
plt.legend()
plt.show()

