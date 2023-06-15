import matplotlib.pyplot as plt
import numpy as np

from .frac_to_cart import frac_to_cart


def plot_jump_paths(sites, lat, show_names):
    # Plot the sites and the jumps between them
    nr_sites = sites['occupancy'].shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the lattice
    for i in range(3):
        ax.plot([0, lat[i, 0]], [0, lat[i, 1]], [0, lat[i, 2]], color='black')
        for j in range(3):
            if i != j:
                ax.plot([lat[i, 0], lat[i, 0] + lat[j, 0]], [lat[i, 1], lat[i, 1] + lat[j, 1]],
                        [lat[i, 2], lat[i, 2] + lat[j, 2]], color='black')
                for k in range(3):
                    if k != i and k != j:
                        ax.plot([lat[i, 0] + lat[j, 0], lat[i, 0] + lat[j, 0] + lat[k, 0]],
                                [lat[i, 1] + lat[j, 1], lat[i, 1] + lat[j, 1] + lat[k, 1]],
                                [lat[i, 2] + lat[j, 2], lat[i, 2] + lat[j, 2] + lat[k, 2]], color='black')

    # Plot the sites (and site-names if show_names is True)
    delta = 0.2  # Small displacement for better text visibility for the names
    for i in range(nr_sites):
        point_size = 15 * np.log(sites['occupancy'][i]) + 1
        if point_size > 1:
            ax.scatter3D(sites['cart_pos'][0, i], sites['cart_pos'][1, i], sites['cart_pos'][2, i],
                         s=point_size, color='blue')
            if show_names:
                ax.text(sites['cart_pos'][0, i] + delta, sites['cart_pos'][1, i] + delta,
                        sites['cart_pos'][2, i] + delta, sites['site_names'][i])

    # Plot the jumps as lines between the sites, and linewidth depending on the number of jumps
    vec_cart = np.zeros((3, 2))
    vec_frac = np.zeros((3, 2))
    vec1 = np.zeros((3, 2))
    vec2 = np.zeros((3, 2))

    for i in range(nr_sites - 1):
        for j in range(i + 1, nr_sites):
            linewidth = 0
            if sites['transitions'][i, j] + sites['transitions'][j, i] > 0:
                linewidth = 1 + np.log(sites['transitions'][i, j] + sites['transitions'][j, i])
            if linewidth > 0:
                color = 'red'
                for k in range(3):
                    vec_cart[k, :] = [sites['cart_pos'][k, i], sites['cart_pos'][k, j]]
                    vec_frac[k, :] = [sites['frac_pos'][k, i], sites['frac_pos'][k, j]]

                pbc = np.array([False, False, False])
                nr_pbc = 0
                for k in range(3):
                    if abs(sites['frac_pos'][k, i] - sites['frac_pos'][k, j]) > 0.5:
                        pbc[k] = True
                        if nr_pbc == 0:
                            nr_pbc = 1
                            index_max = np.argmax(vec_frac[k, :])
                            index_min = np.argmin(vec_frac[k, :])
                            vec1[k, :] = [max(vec_frac[k, :]), min(vec_frac[k, :]) + 1]
                            vec2[k, :] = [max(vec_frac[k, :]) - 1, min(vec_frac[k, :])]
                        else:
                            index_max2 = np.argmax(vec_frac[k, :])
                            if index_max2 == index_max:
                                vec1[k, :] = [max(vec_frac[k, :]), min(vec_frac[k, :]) + 1]
                                vec2[k, :] = [max(vec_frac[k, :]) - 1, min(vec_frac[k, :])]
                            else:
                                vec1[k, :] = [max(vec_frac[k, :]) - 1, min(vec_frac[k, :])]
                                vec2[k, :] = [max(vec_frac[k, :]), min(vec_frac[k, :]) + 1]

                if np.any(pbc):
                    for k in range(3):
                        if not pbc[k]:
                            vec1[k, :] = [vec_frac[k, index_max], vec_frac[k, index_min]]
                            vec2[k, :] = [vec_frac[k, index_max], vec_frac[k, index_min]]

                    vec1a = frac_to_cart(vec1[:, 0], lat)  # Convert to cartesian coordinates
                    vec1b = frac_to_cart(vec1[:, 1], lat)
                    vec2a = frac_to_cart(vec2[:, 0], lat)
                    vec2b = frac_to_cart(vec2[:, 1], lat)

                    ax.plot([vec1a[0], vec1b[0]], [vec1a[1], vec1b[1]], [vec1a[2], vec1b[2]], color=color,
                            linewidth=linewidth)
                    ax.plot([vec2a[0], vec2b[0]], [vec2a[1], vec2b[1]], [vec2a[2], vec2b[2]], color=color,
                            linewidth=linewidth)
                else:
                    ax.plot(vec_cart[0, :], vec_cart[1, :], vec_cart[2, :], color=color, linewidth=linewidth)

    # Set plot properties
    ax.set_title('Jumps between sites')
    ax.set_xlabel('X (Angstrom)')
    ax.set_ylabel('Y (Angstrom)')
    ax.set_zlabel('Z (Angstrom)')
    ax.set_box_aspect([1, 1, 1])
    min_coor = np.sum(lat * (lat < 0), axis=0)
    max_coor = np.sum(lat * (lat > 0), axis=0)
    ax.set_xlim(min_coor[0], max_coor[0])
    ax.set_ylim(min_coor[1], max_coor[1])
    ax.set_zlim(min_coor[2], max_coor[2])
    ax.view_init(-10, 10)

    plt.show()
