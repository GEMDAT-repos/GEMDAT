import matplotlib.pyplot as plt
import numpy as np

from .calc_dist_sqrd_frac import calc_dist_sqrd_frac


def jumps_vs_dist(sites, sim_data, pics, resolution):
    nr_sites = sites['occupancy'].shape[0]
    nr_diffusing = sites['atoms'].shape[1]
    ang2m2 = 1E-20  # Angstrom^2 to meter^2

    lattice = sim_data['lattice']
    dimensions = sim_data['diffusion_dim']
    sim_time = sim_data['total_time']

    max_dist = 10.0  # Much larger than the largest jump distance in typical materials
    jumps_vs_dist = np.zeros(int(np.ceil(max_dist / resolution)))
    max_bin = 0
    jump_diff = 0
    for i in range(nr_sites - 1):
        for j in range(i + 1, nr_sites):
            nr_jumps = sites['transitions'][i, j] + sites['transitions'][j, i]
            if nr_jumps > 0:
                dist = np.sqrt(
                    calc_dist_sqrd_frac(sites['frac_pos'][:, i],
                                        sites['frac_pos'][:, j], lattice))
                # For the jump diffusivity:
                jump_diff += nr_jumps * (dist**2)
                # For the histogram:
                dist_bin = int(np.ceil(dist / resolution))
                jumps_vs_dist[dist_bin] += nr_jumps
                if dist_bin > max_bin:
                    max_bin = dist_bin

    print(
        f'Jump diffusion calculated assuming {dimensions} dimensional diffusion.'
    )
    jump_diff = (jump_diff * ang2m2) / (
        2 * dimensions * nr_diffusing * sim_time)  # In m^2/sec
    print(f'Jump diffusivity (in m^2/sec): {jump_diff:e}')

    # Plot histogram of jumps vs distance of the jump
    if pics:
        distances = np.arange(0.5 * resolution, (max_bin + 0.5) * resolution,
                              resolution)
        plt.figure()
        plt.bar(distances, jumps_vs_dist[:max_bin + 1])
        plt.title('Jumps vs. distance')
        plt.xlabel('Distance (Angstrom)')
        plt.ylabel('Nr. of jumps')

    return jump_diff
