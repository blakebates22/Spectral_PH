"""
This script generates points in the Cantor set up to a given level and calculates a distance
function that represents the distance between any point in [0, 1] and the closest point in
the Cantor set. The script applies a low-pass Butterworth filter to this distance function
with different cutoff frequencies and counts the number of local minima in the filtered signal.
Finally, it plots the number of minima as a function of the cutoff frequency.

Key steps:
1. Generate Cantor set points up to a specified level.
2. Calculate the distance function for each point in the interval [0, 1].
3. Apply a low-pass Butterworth filter to the distance function for different cutoff frequencies.
4. Count the number of local minima in the filtered signal for each cutoff frequency.
5. Plot the relationship between the number of minima and the cutoff frequency (filtration parameter).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema

# Function to generate Cantor set points up to a given level.
# Cantor set is generated recursively by removing the middle third of each interval.
def cantor_set(level, start=0, end=1):
    if level == 0:
        return [start, end]  # Base case: return the endpoints for level 0
    else:
        third = (end - start) / 3  # Calculate the third of the interval
        # Recursively apply the process to the left and right thirds
        return (cantor_set(level - 1, start, start + third) +
                cantor_set(level - 1, end - third, end))

# Function to compute the distance to the closest point in the Cantor set
# for a given value x.
def distance_function(x, cantor_points):
    return min(abs(x - point) for point in cantor_points)  # Calculate the distance to the nearest Cantor point

# Generate Cantor set for 6 levels
level = 6
cantor_points = cantor_set(level)

# Create an array of x values
x_values = np.linspace(0, 1, 1000)

# Calculate the distance function for each x value
distance_values = np.array([distance_function(x, cantor_points) for x in x_values])

# Function to count the number of minima in a filtered signal
def count_minima(filtered_values):
    minima = argrelextrema(filtered_values, np.less)[0]  # Find local minima
    return len(minima)  # Return the count of minima

# Define a range of cutoff frequencies for filtering
cutoff_frequencies = np.linspace(100, 1, 20)  # From 100Hz down to 1Hz

# Sampling frequency
fs = 1000

# List to store the number of minima for each cutoff frequency
minima_counts = []

# Loop over the cutoff frequencies and apply filtering
for cutoff in cutoff_frequencies:
    b, a = butter(N=2, Wn=cutoff/(fs/2), btype='low', analog=False)  # Butterworth filter coefficients
    filtered_distance_values = filtfilt(b, a, distance_values)  # Apply the Butterworth filter

    # Count the number of minima in the filtered function
    minima_count = count_minima(filtered_distance_values)
    minima_counts.append(minima_count)

# Plot the number of minima vs the filtration parameter (cutoff frequency)
plt.figure(figsize=(10, 6))
plt.plot(cutoff_frequencies, minima_counts, marker='o')
plt.gca().invert_xaxis()  # Reverse the x-axis
plt.xlabel('Cutoff Frequency (Hz)')
plt.ylabel('Number of Minima')
plt.title('Number of Minima vs Cutoff Frequency (Filtration Parameter)')
plt.grid(True)
plt.show()
