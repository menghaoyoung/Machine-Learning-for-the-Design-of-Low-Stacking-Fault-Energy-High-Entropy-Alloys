# -*- coding: utf-8 -*-
"""

@author: Administrator
"""
import csv
import random

def generate_composition(ranges):
    # Generate random compositions within the given ranges
    composition = {}
    for element, (min_val, max_val) in ranges.items():
        composition[element] = random.uniform(min_val, max_val)

    return composition

def is_within_range(composition, ranges):
    for element, value in composition.items():
        min_val, max_val = ranges[element]
        if value < min_val or value > max_val:
            return False
    return True

def save_to_csv(data, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write composition data
        writer.writerow(data.values())

# User specified ingredient range
element_ranges = {
    'Fe': (0, 0.5),
    'Mn': (0, 0.3),
    'Ni': (0, 0.78),
    'Co': (0.1, 0.5),
    'Cr': (0.05, 0.35),
    'V': (0, 0.15),
    'Al': (0, 0.06)
}

# Obtain user input
num_samples = int(input("Please enter the number of alloy composition groups to be generated:"))

# Write CSV file header
with open('alloy_compositions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(element_ranges.keys())

# Generate multiple sets of alloy composition data
generated_data = []
while len(generated_data) < num_samples * 10:  # Generate a large amount of data to ensure sufficient quantity after filtering
    composition = generate_composition(element_ranges)
    generated_data.append(composition)

# Normalize the generated data, then filter out the data that meets the criteria and save it to a CSV file
count = 0
for composition in generated_data:
    total_sum = sum(composition.values())
    normalized_composition = {element: value / total_sum for element, value in composition.items()}
    if is_within_range(normalized_composition, element_ranges):
        save_to_csv(normalized_composition, 'alloy_compositions.csv')
        count += 1
        if count >= num_samples:
            break

print("finish")



















            
            
            
        
                
                
       
  
    
    

