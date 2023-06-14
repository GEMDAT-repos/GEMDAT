import numpy as np
import matplotlib.pyplot as plt

def plot_collective(sites):
    ## Plot a histogram of the number of jumps vs. timestep
    centers = np.arange(250, size(sites.atoms[:,0]), 500)
    times = np.sort(sites.all_trans[:,4])
    plt.figure()
    plt.hist(times, bins=centers)
    plt.title('Histogram of jumps vs. time')
    plt.xlabel('Time (steps)')
    plt.ylabel('Nr. of jumps')
    
    ## Plot which types of jumps are 'collective'
    plt.figure()
    plt.imshow(sites.coll_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Number of cooperative jumps per jump-type combination')
    # Put all the jump names on the x- and y-axis
    plt.xticks(np.arange(len(sites.jump_names)), sites.jump_names, rotation=90)
    plt.yticks(np.arange(len(sites.jump_names)), sites.jump_names)