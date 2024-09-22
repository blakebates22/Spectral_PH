import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema

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

# Function to count the number of minima in a filtered signal
def count_minima(filtered_values):
    minima = argrelextrema(filtered_values, np.less)[0]
    return len(minima)

# Define a range of cutoff frequencies for filtering
cutoff_frequencies = np.linspace(100, 1, 20)  # From 50Hz down to 1Hz

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

