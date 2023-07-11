from collections import Counter, defaultdict
from statistics import mean, stdev

from gemdat.sites import _split_transitions_in_parts


def calc_rates(sites, *, n_parts, total_time, n_diffusing, n_steps):
    labels = sites.structure.site_properties['label']
    all_transitions_parts = _split_transitions_in_parts(
        sites.all_transitions, n_steps, n_parts)

    jumps_parts = defaultdict(list)

    for part in all_transitions_parts:
        c = Counter([(labels[i], labels[j]) for i, j in part[:, 1:3]])
        for k, v in c.items():
            jumps_parts[k].append(v)

    rates = {}

    for k, v in jumps_parts.items():
        part_time = total_time / n_parts
        denom = n_diffusing * part_time

        jump_freq_mean = mean(v) / denom
        jump_freq_std = stdev(v) / denom

        rates[k] = jump_freq_mean, jump_freq_std

    return rates


if __name__ == '__main__':
    from gemdat import SimulationData, SitesData, load_known_material

    VASP_XML = '/home/stef/md-analysis-example/vasprun.xml'

    equilibration_steps = 1250
    diffusing_element = 'Li'
    diffusion_dimensions = 3
    z_ion = 1

    data = SimulationData.from_vasprun(VASP_XML, cache='vasprun.xml.cache')

    extras = data.calculate_all(
        equilibration_steps=equilibration_steps,
        diffusing_element=diffusing_element,
        z_ion=z_ion,
        diffusion_dimensions=diffusion_dimensions,
    )

    structure = load_known_material('argyrodite')

    sites = SitesData(structure)
    sites.calculate_all(data=data, extras=extras)

    rates = calc_rates(sites=sites,
                       n_parts=extras.n_parts,
                       total_time=extras.total_time,
                       n_diffusing=extras.n_diffusing,
                       n_steps=extras.n_steps)

    assert isinstance(rates, dict)
    assert len(rates) == 1
    assert rates['Li48h', 'Li48h'] == (188700564971.7514, 16674910449.098799)

    ###

    import numpy as np
    from gemdat.constants import e_charge, k_boltzmann

    labels = sites.structure.site_properties['label']
    n_steps = extras.n_steps
    n_parts = extras.n_parts
    n_diffusing = extras.n_diffusing
    total_time = extras.total_time

    all_transitions_parts = _split_transitions_in_parts(
        sites.all_transitions, n_steps, n_parts)

    jumps_parts = defaultdict(list)

    for part in all_transitions_parts:
        c = Counter([(labels[i], labels[j]) for i, j in part[:, 1:3]])
        for k, v in c.items():
            jumps_parts[k].append(v)

    e_act = {}

    for i, ((site_start, site_stop), v) in enumerate(jumps_parts.items()):
        v = np.array(v)

        part_time = total_time / n_parts

        atom_percentage = sites.atom_locations_parts[i][site_start]

        denom = atom_percentage * n_diffusing * part_time

        eff_rate = v / denom

        if site_start == site_stop:
            eff_rate /= 2

        e_act_arr = -np.log(eff_rate / extras.attempt_freq) * (
            k_boltzmann * data.temperature) / e_charge

        e_act[site_start, site_stop] = np.mean(e_act_arr), np.std(e_act_arr)

    assert isinstance(e_act, dict)
    assert len(e_act) == 1
    assert e_act[('Li48h', 'Li48h')] == (0.14859184952052334,
                                         0.004772810554203923)
