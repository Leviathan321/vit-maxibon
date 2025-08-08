import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def parse_steering_first_element(s):
    s = s.strip("[]")
    parts = s.split()
    return float(parts[0])

def plot_steering_distribution(folder_paths, 
                               csv_filename='actions.csv',
                               normalize = True,
                               title = "Steering Distribution",
                               save_path= None,
                               percentage = 1):
    all_first_elements = []

    for folder_path in folder_paths:
        csv_path = os.path.join(folder_path, csv_filename)
        if not os.path.exists(csv_path):
            print(f"CSV file not found in folder: {folder_path}")
            continue
        df = pd.read_csv(csv_path)
        # Apply parsing to get the first steering element
        df['steering_first'] = df['steering'].apply(parse_steering_first_element)


        steering_list = df['steering_first'].tolist()

        if 0 < percentage < 1.0:
            sample_size = int(len(steering_list) * percentage)
            steering_list = random.sample(steering_list, sample_size)

        all_first_elements.extend(steering_list)

    if not all_first_elements:
        print("No steering data loaded.")
        return

    plt.figure(figsize=(8,5))
    plt.hist(all_first_elements, bins=50, color='green', alpha=0.7, density=normalize)
    plt.xlabel('First Steering Element')
    plt.ylabel('Normalized Frequency' if normalize else 'Frequency')
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path + "_p" + str(percentage) + ".png", format="png")
        print("Image saved in:", save_path)
    else:
        plt.show()
    plt.close()