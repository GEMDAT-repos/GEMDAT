# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:47:32 2023

@author: Matt Damon
"""

import numpy as np  
import matplotlib.pyplot as plt

def bond_direct(traj, cent, sat, nr_cen , nr_lig = 4, norm=True):
    '''returns trajectories of normalized unit vectors defined as the distance
    between a central and satellite atoms, meant to track orientation of
    moleules or clusters'''
    
    # Create new trajectory files
    diff_trajectory_sat = traj.filter(sat)
    diff_trajectory_cent = traj.filter(cent)
    nr_ligands = nr_lig
    nr_central_atoms = nr_cen
    
    #fractional coordinates of central atom and sattelite atoms
    frac_coord_sat = diff_trajectory_sat.positions
    frac_coord_cent = diff_trajectory_cent.positions
    
    
    # Find starting coordinates of all atoms
    central_start_coord = frac_coord_cent[1, :, :]
    sattelite_start_coord = frac_coord_sat[1, :, :]
    
    # Calculate distances between every central atom and all satellite atoms
    lattice = traj.lattice
    
    #function to calculate distance between two positions taking BC intro account
    def calc_dist(frac1, frac2, lattice):
        frac = np.subtract(frac2, frac1)
        
        # Use numpy's modulo operation to handle wrapping
        frac = np.mod(frac + 0.5, 1) - 0.5
    
        cart = np.dot(frac, lattice)
        dist = np.linalg.norm(cart)
        return dist
    
    distance = np.array([[calc_dist(central, satellite, lattice) for satellite in sattelite_start_coord] for central in central_start_coord])
    
    # Determine which satellite atoms are close enough to central atom to be connected
    match_criteria = 1.5 * np.min(distance)
    distance_match = np.where(distance < match_criteria, distance, 0)
    
    # Create array which contains every central atom at rows and associated satellite atoms in the columns
    matching_matrix = np.zeros((len(frac_coord_cent[0, :, 0]), 4), dtype=int)
    
    for k in range(len(frac_coord_cent[0, :, 0])):
        matching_matrix[k, :] = np.where(distance_match[k, :] != 0)[0][:4]
    
    # Get central atoms and satellite atoms in a nice matrix for further calculations
    index_central_atoms = np.arange(nr_central_atoms)
    combinations = np.array([(i, j) for i in index_central_atoms for j in matching_matrix[i, :]])
    
    # Get all fractional coordinate vectors going from central atom to its 4 ligands
    direction = frac_coord_sat[:, combinations[:, 1], :] - frac_coord_cent[:, combinations[:, 0], :]
    
    # Take the periodic boundary conditions into account
    direction = np.where(direction > 0.5, direction - 1, direction)
    direction = np.where(direction < -0.5, direction + 1, direction)
    
    # Create a directions matrix with cartesian vectors
    # direct_cart = np.matmul(lattice.T, np.swapaxes(direction, 2, 1))
    direct_cart = np.matmul(direction, lattice)
    
    if norm:
        normalized_direct_cart = direct_cart / np.linalg.norm(direct_cart, axis=-1, keepdims=True)
        return normalized_direct_cart
    else:
        return direct_cart

def cart2sph(x, y, z):
    '''transform cartesian coordinates to spherical coordinates'''
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.arcsin(z / r)
    az = np.arctan2(y, x)
    return az, el, r

def cartesian_to_spherical(direct_cart, degrees = True):
   '''trajectory from cartesian coordinates to spherical coordinates'''
   x = direct_cart[:, :, 0]
   y = direct_cart[:, :, 1]
   z = direct_cart[:, :, 2]
    
   az, el, r = cart2sph(x, y, z)
    
   if degrees:
        az = np.degrees(az)
        el = np.degrees(el)
    
   # Stack the results along the last axis to match the shape of direction_spherical
   direction_spherical = np.stack((az, el, r), axis=-1)
    
   return direction_spherical

def sph_prob(direction_spherical_deg, shape =(360,180)):

    az = direction_spherical_deg[:,:,0].flatten()
    el = direction_spherical_deg[:,:,1].flatten()
    
    # Compute the 2D histogram - for reasons, x-y inversed
    hist, xedges, yedges = np.histogram2d(el, az, shape)
    
    
    def calculate_spherical_areas(shape, radius=1):
        azimuthal_angles = np.linspace(0, 360, shape[0])
        elevation_angles = np.linspace(0, 180, shape[1])

        areas = np.zeros(shape, dtype=float)

        for i in range(shape[0]):
            for j in range(shape[1]):
                azimuthal_increment = np.deg2rad(1)
                elevation_increment = np.deg2rad(1)

                areas[i, j] = (radius**2) * azimuthal_increment * np.sin(np.deg2rad(elevation_angles[j])) * elevation_increment
                #hacky way to get rid of singularity on poles
                areas[:,0] = areas[:,-1]
        return areas
    
    areas= calculate_spherical_areas(shape)
        
    hist = np.divide(hist, areas)
    
    #replace values at the poles where normalization breaks - hacky
    hist[:,0]=hist[:,1]
    hist[:,-1]=hist[:,-2]
    return hist

def rectilinear_plot(grid):
    '''plot a rectilinear projection of a spherical function'''
    values = grid.T
    phi = np.linspace(0, 360, np.ma.size(values,0))
    theta = np.linspace(0, 180, np.ma.size(values,1))
     
    theta, phi = np.meshgrid(theta,phi)
    
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='rectilinear'))
    cs= ax.contourf(phi, theta, values, cmap='viridis')
    ax.set_yticks(np.arange(0,190,45))
    ax.set_xticks(np.arange(0,370,45))  
    
    ax.set_xlabel(r'azimuthal angle φ $[\degree$]')
    ax.set_ylabel(r'elevation θ $[\degree$]')
    
    ax.grid(visible=True)
    cbar = fig.colorbar(cs, label= 'areal probability', format='')
    
    # Rotate the colorbar label by 180 degrees
    cbar.ax.yaxis.set_label_coords(2.5, 0.5)  # Adjust the position of the label
    cbar.set_label('areal probability', rotation=270, labelpad=15)
    return 