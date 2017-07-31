#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:51:02 2017

@author: David
"""

import matplotlib.pyplot as plt

# Create a 2-dimensional array of iris characteristics
def parse_data(lines, num_lines):
    # To store each characteristic of an iris
    data = [[], [], [], [], []]
    # For each line
    for i in range(num_lines):
        # Break up the line
        line = lines[i].split(',')
        # For each list in data
        for j in range(len(data)):
            # Add each component of the line to each
            # sub-array in data
            data[j].append(line[j])
    return data


with open('iris.csv') as f:
    # Store 2d array of iris characteristics
    flowers = parse_data(f.readlines(), 40)

plt.plot(flowers[0])

plt.xlabel('Flowers')
plt.ylabel('Sepal Length')
plt.title('Sepal Length of Flowes in Iris')
plt.grid(True)
plt.savefig("test.png", 500, 500)
plt.show()
