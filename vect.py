import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_even_vectors_3d(N):
    vectors = []
    
    # Iterate to generate N points on the surface of a sphere
    for i in range(N):
        theta = np.arccos(1 - 2 * (i + 0.5) / N)  # Polar angle
        phi = np.pi * (1 + 5 ** 0.5) * i  # Azimuthal angle, using golden angle
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        vectors.append([x, y, z])
    
    return vectors

labels = generate_even_vectors_3d(10)
print(labels)