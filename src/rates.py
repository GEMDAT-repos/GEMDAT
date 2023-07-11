from collections import Counter, defaultdict
from statistics import mean, stdev

from gemdat.sites import _split_transitions_in_parts


def calc_rates(sites, *, n_parts, total_time, n_diffusing, n_steps):
    labels = sites.structure.site_properties['label']
    all_transitions_parts = _split_transitions_in_parts(
        sites.all_transitions, n_steps, n_parts)

    counts_parts = defaultdict(list)

    for part in all_transitions_parts:
        c = Counter([(labels[i], labels[j]) for i, j in part[:, 1:3]])
        for k, v in c.items():
            counts_parts[k].append(v)

    rates = {}

    for k, v in counts_parts.items():
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
    assert rates[('Li48h', 'Li48h')] == (188700564971.7514, 16674910449.098799)
