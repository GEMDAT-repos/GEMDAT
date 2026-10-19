from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from pymatgen.core import Element, Species, Structure
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.stats import skewnorm

from gemdat.symmetry import SymmetryAnalyzer, _site_images

if TYPE_CHECKING:
    from gemdat.crystallizer import CrystallizerScan
    from gemdat.jumps import Jumps
    from gemdat.orientations import Orientations
    from gemdat.trajectory import Trajectory
    from gemdat.volume import Volume


def _mean_displacements_per_element(
    trajectory: Trajectory,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calculate mean displacements per element type.

    Helper for `displacement_per_atom`.
    """
    species = trajectory.species

    grouped = defaultdict(list)
    for sp, distances in zip(species, trajectory.distances_from_base_position()):
        assert isinstance(sp, (Species, Element)), f'got {type(sp)}'

        grouped[sp.symbol].append(distances)

    means = {}
    for sp, dists in grouped.items():
        mean = np.mean(dists, axis=0)
        std = np.std(dists, axis=0)

        means[sp] = (mean, std)

    return means


def _orientations_to_histogram(
    orientations: Orientations, /, *, bins: int = 50
) -> pd.DataFrame:
    """Calculate bond lenth histogram from `Orientations`

    Helper for `bond_length_distribution`.
    """
    *_, bond_lengths = orientations.vectors_spherical.T
    bond_lengths = bond_lengths.flatten()

    hist, edges = np.histogram(bond_lengths, bins=bins, density=True)
    bin_centers = bin_centers = (edges[:-1] + edges[1:]) / 2

    return pd.DataFrame(
        {
            'prob': hist,
            'center': bin_centers,
            'left_edge': edges[:-1],
            'right_edge': edges[1:],
        }
    )


def _fit_skewnorm_to_hist(
    df: pd.DataFrame, /, steps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit skewnorm to bond lenth histogram from `Orientations`

    Helper for `bond_length_distribution`.
    """
    bin_centers = df['center']
    hist = df['prob']

    params, covariance = curve_fit(skewnorm.pdf, bin_centers, hist, p0=[1.5, 1, 1.5])

    x = np.linspace(min(bin_centers), max(bin_centers), steps)
    y = skewnorm.pdf(x, *params)

    return x, y


def hex2rgba(hex_color: str, *, opacity: float = 1) -> str:
    """Convert hex string to rgba."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    return f'rgba({r},{g},{b},{opacity})'


def density_matching_cmap_mpl():
    colors = ['white', 'orangered', 'greenyellow', 'cyan']
    isovals = [0.0, 0.65, 0.30, 0.05]
    alphavals = [0.0, 0.5, 1, 0.5]

    # Sort isovals along with colors and alphavals
    sorted_indices = np.argsort(isovals)
    isovals = np.array(isovals)[sorted_indices]
    colors = np.array(colors)[sorted_indices]
    alphavals = np.array(alphavals)[sorted_indices]

    # Normalize isovals to create positions for the colormap
    norm_isovals = isovals / max(isovals)

    # Ensure the first and last points are at 0 and 1
    if norm_isovals[0] != 0:
        norm_isovals = np.insert(norm_isovals, 0, 0)
        colors = np.insert(colors, 0, colors[0])
        alphavals = np.insert(alphavals, 0, alphavals[0])

    if norm_isovals[-1] != 1:
        norm_isovals = np.append(norm_isovals, 1)
        colors = np.append(colors, colors[-1])
        alphavals = np.append(alphavals, alphavals[-1])

    # Create a custom colormap
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    for iso, color, alpha in zip(norm_isovals, colors, alphavals):
        r, g, b = to_rgb(color)
        cdict['red'].append((iso, r, r))
        cdict['green'].append((iso, g, g))
        cdict['blue'].append((iso, b, b))
        cdict['alpha'].append((iso, alpha, alpha))

    custom_cmap = LinearSegmentedColormap('custom_cmap', segmentdata=cdict, N=256)
    return custom_cmap


def density_matching_colorscale_plotly() -> list[tuple[float, str]]:
    """Plotly colorscale equal to `density_matching_cmap_mpl`."""
    cmap = density_matching_cmap_mpl()
    return [
        (
            float(pos),
            'rgba({:.0f},{:.0f},{:.0f},{:.3f})'.format(
                *np.multiply(cmap(pos), 255)[:3], cmap(pos)[3]
            ),
        )
        for pos in np.linspace(0, 1, 32)
    ]


@dataclass
class VibrationalAmplitudeHist:
    amplitudes: np.ndarray
    counts: np.ndarray
    std: np.ndarray

    @property
    def min_amp(self):
        return self.amplitudes.min()

    @property
    def max_amp(self):
        return self.amplitudes.max()

    @property
    def width(self):
        bins = len(self.amplitudes)
        return (self.max_amp - self.min_amp) / bins

    @property
    def offset(self):
        return self.width / 2

    @property
    def centers(self):
        return self.amplitudes + self.offset

    @property
    def dataframe(self):
        return pd.DataFrame(
            data=zip(self.centers, self.counts, self.std), columns=['center', 'count', 'std']
        )


def _get_vibrational_amplitudes_hist(
    *, trajectories: list[Trajectory], bins: int
) -> VibrationalAmplitudeHist:
    """Calculate vibrational amplitudes histogram.

    Helper for `vibrational_amplitudes`.
    """
    metrics = [trajectory.metrics().amplitudes() for trajectory in trajectories]

    max_amp = max(max(metric) for metric in metrics)
    min_amp = min(min(metric) for metric in metrics)

    max_amp = max(abs(min_amp), max_amp)
    min_amp = -max_amp

    data = []

    for metric in metrics:
        data.append(np.histogram(metric, bins=bins, range=(min_amp, max_amp), density=True)[0])

    amplitudes = np.linspace(min_amp, max_amp, bins, endpoint=False)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    return VibrationalAmplitudeHist(amplitudes=amplitudes, counts=mean, std=std)


def _jumps_vs_distance(jumps: Jumps, *, resolution: float, n_parts: int) -> pd.DataFrame:
    """Calculate jumps vs distance histogram.

    Helper for `jumps_vs_distance`.
    """
    sites = jumps.sites
    trajectory = jumps.trajectory
    lattice = trajectory.get_lattice()

    pdist = lattice.get_all_distances(sites.frac_coords, sites.frac_coords)

    bin_max = (1 + pdist.max() // resolution) * resolution
    n_bins = int(bin_max / resolution) + 1
    x = np.linspace(0, bin_max, n_bins)

    bin_idx = np.digitize(pdist, bins=x)
    data = []
    for transitions_part in jumps.split(n_parts=n_parts):
        counts = np.zeros_like(x)
        for idx, n in zip(bin_idx.flatten(), transitions_part.matrix().flatten()):
            counts[idx] += n
        for idx in range(n_bins):
            if counts[idx] > 0:
                data.append((x[idx], counts[idx]))

    df = pd.DataFrame(data=data, columns=['Displacement', 'count'])

    grouped = df.groupby(['Displacement'])
    mean = grouped.mean().reset_index().rename(columns={'count': 'mean'})
    std = grouped.std().reset_index().rename(columns={'count': 'std'})
    df = mean.merge(std, how='inner')

    return df


def _specie_sites(scan: CrystallizerScan, specie: str | None) -> tuple[str, list[int]]:
    """The species (default: the mobile one) and the indices of its sites."""
    if scan.crystallizer is None:
        raise ValueError('This scan carries no density; make it with `Crystallizer.scan()`.')
    specie = specie or scan.crystallizer.floating_specie
    indices = [i for i, site in enumerate(scan.geometry) if site.specie.symbol == specie]
    if not indices:
        raise ValueError(f'The scan has no {specie} sites.')
    return specie, indices


def _crystallizer_fit(scan: CrystallizerScan, symprec: float | None):
    """Symmetry analyzer, tolerance and spglib dataset of the fit to plot."""
    symprec = scan.ranking.best().symprec if symprec is None else symprec
    analyzer = SymmetryAnalyzer(scan.geometry, angle_tolerance=scan.angle_tolerance)
    return analyzer, symprec, analyzer._dataset(symprec)


def _crystallizer_sites(
    scan: CrystallizerScan, symprec: float | None, specie: str | None
) -> tuple[str, Structure, Volume]:
    """Data for `crystallizer_sites`.

    Returns the title, the idealised sites of `specie` (default: the
    mobile species) with their occupancies and labelled by orbit
    (Wyckoff letter and multiplicity), and the density of `specie`.
    """
    specie, indices = _specie_sites(scan, specie)
    density = scan.density(specie)
    analyzer, symprec, dataset = _crystallizer_fit(scan, symprec)
    ideal = analyzer.idealise(symprec).frac_coords % 1.0
    orbits = np.array(dataset.equivalent_atoms)

    labels = []
    for i in indices:
        multiplicity = (orbits == orbits[i]).sum()
        labels.append(f'{specie} {dataset.wyckoffs[orbits[i]]} ({multiplicity})')
    sites = Structure(
        lattice=scan.geometry.lattice,
        species=[{specie: scan.occupancies[i]} for i in indices],
        coords=ideal[indices],
        labels=labels,
    )

    title = f'{dataset.international} (#{dataset.number}) at symprec={symprec:.3g} Å'
    return title, sites, density


def _crystallizer_site_images(
    scan: CrystallizerScan,
    symprec: float | None,
    specie: str | None,
    site: int | None,
    width: float,
    resolution: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Data for `crystallizer_site_images`.

    Returns the title, the box axis ticks (Å), the density of the site's
    species sampled on a Cartesian box around the ideal site, and the
    images in Cartesian coordinates (Å) relative to the ideal site.
    """
    if site is None:
        site = _specie_sites(scan, specie)[1][0]
    density = scan.density(scan.geometry[site].specie.symbol)
    analyzer, symprec, dataset = _crystallizer_fit(scan, symprec)
    lattice = scan.geometry.lattice

    actual = scan.geometry.frac_coords[site]
    ideal = analyzer.idealise(symprec).frac_coords[site]
    ideal = ideal - np.round(ideal - actual)
    center = lattice.get_cartesian_coords(ideal)
    images = _site_images(scan.geometry, dataset)[:, site]
    images = lattice.get_cartesian_coords(images) - center

    # Sample the density on a Cartesian box around the site, interpolating
    # linearly between voxel centres ((i + 1/2) / n) under periodic boundaries.
    ticks = np.arange(-width, width + resolution / 2, resolution)
    grid = np.stack(np.meshgrid(ticks, ticks, ticks, indexing='ij'), axis=-1)
    frac = lattice.get_fractional_coords(grid.reshape(-1, 3) + center) % 1.0
    voxels = frac * density.dims - 0.5
    box = map_coordinates(density.data, voxels.T, order=1, mode='grid-wrap')
    box = box.reshape(grid.shape[:3])

    # Label the site by its orbit, as `crystallizer_sites` does.
    orbits = np.array(dataset.equivalent_atoms)
    multiplicity = (orbits == orbits[site]).sum()
    title = (
        f'{scan.geometry[site].specie} {dataset.wyckoffs[orbits[site]]} ({multiplicity}) in '
        f'{dataset.international} (#{dataset.number}) at symprec={symprec:.3g} Å'
    )
    return title, ticks, box, images
