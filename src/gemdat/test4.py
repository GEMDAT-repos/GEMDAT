import matplotlib.pyplot as plt
from gemdat import Trajectory
import numpy as np
import time
import gemdat_rotation as gr

#%%
start_time= time.time()
traj = Trajectory.from_vasprun("vasprun.xml")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Loaded .xml in  {elapsed_time} seconds")

#%%

normalized_direct_cart = gr.bond_direct(traj, "S", "O",  8)

#%%
# Matrix to transform primitive unit cell coordinates to conventional unit cell coordinates
prim_to_conv_matrix = np.array([[1/np.sqrt(2), -1/np.sqrt(6), 1/np.sqrt(3)],
                                [1/np.sqrt(2), 1/np.sqrt(6), -1/np.sqrt(3)],
                                [0, 2/np.sqrt(6), 1/np.sqrt(3)]])

# Transfrom direct_cart to conventional form
direct_cart_TRANS = np.matmul(normalized_direct_cart, prim_to_conv_matrix.T)

#%%

# Define the symmetry operations for the Oh point group
sym_ops_Oh_firstcolumn = np.array([
    np.eye(3),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
])

sym_ops_Oh_firstrow = np.array([
    np.eye(3),
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
])




# Multiply row by column to get 8x6 matrix of transformation matrices
sym_ops_Oh = np.zeros((8,6,3,3))

for i in range(len(sym_ops_Oh_firstcolumn)):
    for j in range(len(sym_ops_Oh_firstrow)):
        sym_ops_Oh[i,j,:,:] = np.matmul(sym_ops_Oh_firstcolumn[i,:,:], sym_ops_Oh_firstrow[j,:,:])

# sym_ops_Oh = sym_ops_Oh.reshape(3, 3, -1)
#%%
# Reshape to make it easier to iterate on (only on 3rd dimension)
sym_ops_Oh = sym_ops_Oh.reshape(3, 3, -1)
#%%
# Apply symmetry elements of Oh space group with vectorized operations
direction_cart = direct_cart_TRANS[1000:2000,:,:]

start_time= time.time()

n_ts = direction_cart.shape[0]
n_bonds = direction_cart.shape[1]
n_symops = sym_ops_Oh.shape[2]

direction_cart_symOh=np.zeros((n_ts,n_bonds * n_symops,3))
for m in range(n_ts):   
    for l in range(n_bonds):
        for k in range(n_symops):
            direction_cart_symOh[m,l*n_symops+k,:] = np.matmul(sym_ops_Oh[:,:,k], direction_cart[m,l,:])


# direction_cart_symOh =  direction_cart_symOh.reshape(n_ts,-1,3)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"symmetrized direct cart  {elapsed_time} seconds")

#%%

m=0
l=30
k=46

test = np.matmul(sym_ops_Oh[:,:,k], direction_cart[m,l,:])

print(test)

print(direction_cart_symOh[m,(l+1)*(k+1)-1,:])


#%%
direction_spherical_deg = gr.cartesian_to_spherical(direction_cart_symOh, degrees=True)

import gemdat_rotation as gr

shape=(90,360)
az = direction_spherical_deg[:,:,0].flatten()
el = direction_spherical_deg[:,:,1].flatten()

# Compute the 2D histogram - for reasons, x-y inversed
hist, xedges, yedges = np.histogram2d(el, az , shape)


def calculate_spherical_areas(shape, radius=1):
    azimuthal_angles = np.linspace(0, 360, shape[1])
    elevation_angles = np.linspace(0, 180, shape[0])

    areas = np.zeros(shape, dtype=float)

    for i in range(shape[1]):
        for j in range(shape[0]):
            azimuthal_increment = np.deg2rad(1)
            elevation_increment = np.deg2rad(1)

            areas[j,i] = (radius**2) * azimuthal_increment * np.sin(np.deg2rad(elevation_angles[j])) * elevation_increment
            #hacky way to get rid of singularity on poles
            areas[0,:] = areas[-1,0]
    return areas

areas= calculate_spherical_areas(shape)
    
h_norm = np.divide(hist, areas)

#replace values at the poles where normalization breaks - hacky
h_norm[0,:] = h_norm[1,:]
h_norm[-1,:] = h_norm[-2,:]


# hist = gr.sph_prob(direction_spherical_deg)
plt.close('all')
gr.rectilinear_plot(hist)
gr.rectilinear_plot(h_norm)


#%%
plt.figure()
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

r = direction_spherical_deg[:,:,2].flatten()
data = r

# Specify the number of bins
bins = 1000

# Plot the normalized histogram
hist, edges = np.histogram(data, bins=bins, density=True)
bin_centers = (edges[:-1] + edges[1:]) / 2

# Fit a skewed Gaussian distribution to the data
def skewed_gaussian(x, loc, scale, skew):
    return skewnorm.pdf(x, a=skew, loc=loc, scale=scale)

params, covariance = curve_fit(skewed_gaussian, bin_centers, hist, p0=[1.5, 1, 1.5])

# Plot the histogram
plt.hist(data, bins=bins, density=True, color='blue', alpha=0.7, label='Data')

# Plot the fitted skewed Gaussian distribution
x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
plt.plot(x_fit, skewed_gaussian(x_fit, *params), 'r-', label='Skewed Gaussian Fit')

plt.xlabel(r'bond length $[\AA]$')
plt.ylabel(r'probability density $[\AA^{-1}]$')
plt.title('Bond Length Probability Distribution')
plt.legend()
plt.grid(True)
plt.show()
#%%
#calculate vector autocorr

def autocorr(direct_cart, Npt=100):
    start_time = time.time()
    
    Nts=len(normalized_direct_cart) # number timesteps
    ep=Nts-1    
    dts= np.round(np.logspace(1, np.log10(ep), Npt)).astype(int)
    
    mean_autocorr=np.zeros(len(dts))
    
    for k, dt in enumerate(dts):
        autocorr = np.sum(normalized_direct_cart[:-dt] * normalized_direct_cart[dt:], axis=-1)
        mean_autocorr[k] = np.mean(autocorr)
    
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Autocorrelation function calculated in  {elapsed_time} seconds")
    
    return dts*0.002, mean_autocorr #time in ps

x,y = autocorr(normalized_direct_cart)

plt.scatter(x,y)
#%%\
    
def autocorr_par(direct_cart, Npt=100):
    from concurrent.futures import ThreadPoolExecutor
    
    # Assuming normalized_direct_cart is already defined
    
    def compute_autocorr(dt):
        autocorr = np.sum(normalized_direct_cart[:-dt] * normalized_direct_cart[dt:], axis=-1)
        return np.mean(autocorr)
    
    start_time = time.time()
    
    Nts = len(normalized_direct_cart)
    ep = Nts - 1
    dts = np.round(np.logspace(1, np.log10(ep), Npt)).astype(int)
    
    mean_autocorr = np.zeros(len(dts))
    
    with ThreadPoolExecutor() as executor:
        mean_autocorr = list(executor.map(compute_autocorr, dts))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Autocorrelation function calculated in {elapsed_time} seconds")
    
    return dts*0.002, mean_autocorr

x,y = autocorr_par(normalized_direct_cart)

plt.scatter(x,y)

#%%

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def autocorr_par(direct_cart, Npt=100):
    def compute_autocorr(dt):
        autocorr = np.sum(direct_cart[:-dt] * direct_cart[dt:], axis=-1)
        mean_autocorr = np.mean(autocorr)
        std_autocorr = np.std(autocorr)
        return mean_autocorr, std_autocorr
    
    Nts = len(direct_cart)
    ep = Nts - 1
    dts = np.round(np.logspace(1, np.log10(ep), Npt)).astype(int)
    
    mean_autocorr = np.zeros(len(dts))
    std_autocorr = np.zeros(len(dts))
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_autocorr, dts))
    
    for i, (mean, std) in enumerate(results):
        mean_autocorr[i] = mean
        std_autocorr[i] = std
    
    return dts * 0.002, mean_autocorr, std_autocorr

x, y, y_std = autocorr_par(normalized_direct_cart)

# Plot the mean autocorrelation
plt.scatter(x, y, label='Mean Autocorrelation')

# Plot error bars using standard deviation
plt.errorbar(x, y, yerr=y_std, fmt='o', label='Mean Autocorrelation with Std Dev')

plt.xlabel('Time Lag (ps)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation with Standard Deviation')
plt.legend()
plt.grid(True)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def autocorr_par(direct_cart, Npt=100):
    def compute_autocorr(dt):
        autocorr = np.sum(direct_cart[:-dt] * direct_cart[dt:], axis=-1)
        mean_autocorr = np.mean(autocorr)
        sem_autocorr = np.std(autocorr) / np.sqrt(len(autocorr))
        return mean_autocorr, sem_autocorr
    
    Nts = len(direct_cart)
    ep = Nts - 50
    dts = np.round(np.logspace(1, np.log10(ep), Npt)).astype(int)
    
    mean_autocorr = np.zeros(len(dts))
    sem_autocorr = np.zeros(len(dts))
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_autocorr, dts))
    
    for i, (mean, sem) in enumerate(results):
        mean_autocorr[i] = mean
        sem_autocorr[i] = sem
    
    return dts * 0.002, mean_autocorr, sem_autocorr

x, y, y_sem = autocorr_par(normalized_direct_cart, Npt=100)

# Plot the mean autocorrelation
#plt.scatter(x, y, label='Mean Autocorrelation')

# Plot error bars using standard error
plt.errorbar(x, y, yerr=y_sem, fmt='o', label='Mean Autocorrelation with Std Error')
# plt.errorbar(x, y, yerr=y_sem, fmt='s', markersize=0, label='Mean Autocorrelation with Std Error')

plt.xlabel('Time Lag (ps)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation with Standard Error')
plt.legend()
plt.grid(True)
plt.show()
