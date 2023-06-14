import numpy as np
import matplotlib.pyplot as plt

def plot_rdfs(rdf):
    # Plot all the rdf's
    nr_rdfs = len(rdf['rdf_names'])
    for i in range(nr_rdfs):
        if rdf['integrated'][0,-1,i] > 0:
            # Density vs. distance
            plt.figure()
            plt.plot(rdf['distributions'][:,:,i].T)
            plt.title(rdf['rdf_names'][i])
            plt.xticks(np.arange(rdf['max_dist']+1))
            plt.xlabel('Distance (Angstrom)')
            plt.ylabel('Density (a.u)')
            plt.legend(rdf['elements'])
            plt.grid(True)
            # The INTEGRATED density vs. distance:
            #plt.figure()
            #plt.plot(rdf['integrated'][:,:,i].T)
            #plt.title(rdf['rdf_names'][i])
            #plt.xticks(np.arange(rdf['max_dist']+1))
            #plt.xlabel('Distance (Angstrom)')
            #plt.ylabel('Density (a.u)')
            #plt.legend(rdf['elements'])
            #plt.grid(True)
