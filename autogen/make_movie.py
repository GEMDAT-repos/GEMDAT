import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def make_movie(sites, sim_data, start_end, movie_file, step_frame):
    # Make a movie showing how the atoms jump from site to site
    # LATTICE NOT TAKEN INTO ACCOUNT CORRECTLY YET!

    sim_data['nr_steps']
    nr_atoms = sim_data['nr_diffusing']
    F = FFMpegWriter(movie_file, codec='libx264')
    F.setup(None, fps=24)

    # FOR SOME STRANGE REASON SHOWING THE MOVIE TO THE SCREEN IS ~4 TIMES
    # FASTER AS NOT SHOWING IT...
    # fig = plt.figure(figsize=(8, 8), dpi=80)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # The look of the figure:
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
    ax.view_init(-10, 10)
    ax.grid(True)

    # Initialize some stuff:
    color_vec = plt.cm.hsv(range(nr_atoms))
    line_color = 'red'
    linewidth = 3.0
    pos_size = 200
    atom_size = 170

    print(
        'Making a movie of the jumps, will take some time (depending on the number of jumps)...'
    )
    print(f'Number of time steps per frame: {step_frame}')

    # Get frames at which a jump occurs and where it is in all_trans
    all_jump_times, trans_nr = zip(
        *sorted(zip(sites['all_trans'][:, 4], range(len(sites['all_trans'])))))
    all_jumps = len(trans_nr)
    start_jump = 0
    end_jump = 0
    for i in range(all_jumps):
        if all_jump_times[i] > start_end[0] and start_jump == 0:
            start_jump = i
        elif all_jump_times[i] > start_end[1] and end_jump == 0:
            end_jump = i
    if start_jump == 0:
        print(
            'ERROR! No jumps between the given time steps, not making a movie!'
        )
        return
    elif end_jump == 0:
        end_jump = all_jumps
    jump_times = all_jump_times[start_jump:end_jump]
    nr_jumps = len(jump_times)

    sites['frac_pos'].shape[1]
    lines1, lines2 = None, None

    def update_frame(i):
        nonlocal lines1, lines2

        if i == start_end[0]:
            # Draw the positions
            ax.scatter3D(sites['frac_pos'][0, :],
                         sites['frac_pos'][1, :],
                         sites['frac_pos'][2, :],
                         s=pos_size,
                         edgecolors='black',
                         linewidths=1)
            # The atoms
            atoms = ax.scatter3D(sites['frac_pos'][0, :],
                                 sites['frac_pos'][1, :],
                                 sites['frac_pos'][2, :],
                                 s=atom_size,
                                 c=color_vec,
                                 cmap='hsv')

            F.grab_frame()
            return atoms

        jump_count = 0
        disp_count = 0

        while jump_count < nr_jumps and jump_times[jump_count] == i:
            sites['all_trans'][trans_nr[jump_count], 0]
            prev_site = sites['all_trans'][trans_nr[jump_count], 1]
            site_nr = sites['all_trans'][trans_nr[jump_count], 2]
            prev_pos = sites['frac_pos'][:, prev_site]

            atoms._offsets3d = (sites['frac_pos'][0, :],
                                sites['frac_pos'][1, :],
                                sites['frac_pos'][2, :])
            atoms._facecolor3d = color_vec

            new_pos = sites['frac_pos'][:, site_nr]
            vec1, vec2 = line_pbc(prev_pos, new_pos)

            if vec2.any():
                lines1 = ax.plot3D(vec1[0, :],
                                   vec1[1, :],
                                   vec1[2, :],
                                   color=line_color,
                                   linewidth=linewidth)
                lines2 = ax.plot3D(vec2[0, :],
                                   vec2[1, :],
                                   vec2[2, :],
                                   color=line_color,
                                   linewidth=linewidth)
            else:
                lines1 = ax.plot3D(vec1[0, :],
                                   vec1[1, :],
                                   vec1[2, :],
                                   color=line_color,
                                   linewidth=linewidth)
                lines2 = ax.plot3D([0, 0], [0, 0], [0, 0],
                                   color=line_color,
                                   linewidth=1E-10)

            jump_count += 1

        if i % step_frame == 0:
            disappear_speed = 0.0625 - 1E-10 / 48
            for j in range(disp_count, jump_count):
                if lines1[j].get_linewidth() > 1E-9:
                    lines1[j].set_linewidth(lines1[j].get_linewidth() -
                                            disappear_speed)
                else:
                    disp_count += 1
                if lines2[j].get_linewidth() > 1E-9:
                    lines2[j].set_linewidth(lines2[j].get_linewidth() -
                                            disappear_speed)

            ax.set_title(f'Timestep {i}')
            F.grab_frame()

        return atoms

    anim = plt.FuncAnimation(fig,
                             update_frame,
                             frames=range(start_end[0], start_end[1] + 1),
                             interval=200)
    anim.save(movie_file, writer=F)

    print('Movie done')


def line_pbc(pos1, pos2):
    # Plot a line between pos1 and pos2, also take care of PBC
    vec = [pos1, pos2]
    pbc = [False] * 3
    vec1 = [[0, 0], [0, 0], [0, 0]]
    vec2 = [[0, 0], [0, 0], [0, 0]]

    nr_pbc = 0
    for k in range(3):
        if abs(vec[k][0] - vec[k][1]) > 0.5:
            pbc[k] = True
            if nr_pbc == 0:
                nr_pbc = 1
                index_max = max(range(2), key=lambda x: vec[k][x])
                index_min = min(range(2), key=lambda x: vec[k][x])
                vec1[k] = [vec[k][index_max], vec[k][index_min] + 1]
                vec2[k] = [vec[k][index_max] - 1, vec[k][index_min]]
            else:
                index_max2 = max(range(2), key=lambda x: vec[k][x])
                if index_max2 == index_max:
                    vec1[k] = [vec[k][index_max], vec[k][index_min] + 1]
                    vec2[k] = [vec[k][index_max] - 1, vec[k][index_min]]
                else:
                    vec1[k] = [vec[k][index_max] - 1, vec[k][index_min]]
                    vec2[k] = [vec[k][index_max], vec[k][index_min] + 1]

    if any(pbc):
        for k in range(3):
            if not pbc[k]:
                vec1[k] = [vec[k][index_max], vec[k][index_min]]
                vec2[k] = [vec[k][index_max], vec[k][index_min]]
    else:
        vec1 = vec

    return vec1, vec2
