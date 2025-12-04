import numpy as np
from scipy.spatial.distance import cdist
import sys

# Define file paths
base_path = "/project/MCChu/khluo/Downloads/vide_public/void_vide/1024/{}/z_{}/snapshot_1024/output/sim_ss0.01/sample_sim_ss0.01_z{}_d00"
case = sys.argv[1]
redshift = sys.argv[2]
formatted_redshift = "{:.2f}".format(float(redshift))
input_file = f"{base_path.format(case, redshift, formatted_redshift)}/match_all_particles.npy"

# Define radius bins (in h^-1 Mpc)
bin_ranges = [(10.0, 25.0)]  # First bin: 10-15, second bin: 10-25

# Load data
data = np.load(input_file)

# Extract unique void information
voidIDs_unique = np.unique(data['voidID'])
void_centers_list = []
for voidID in voidIDs_unique:
    mask = data['voidID'] == voidID
    center_x = data['center_x'][mask][0]
    center_y = data['center_y'][mask][0]
    center_z = data['center_z'][mask][0]
    radius = data['radius'][mask][0]
    void_avg_vx = data['void_avg_vx'][mask][0]
    void_avg_vy = data['void_avg_vy'][mask][0]
    void_avg_vz = data['void_avg_vz'][mask][0]
    void_centers_list.append((voidID, center_x, center_y, center_z, radius, void_avg_vx, void_avg_vy, void_avg_vz))

void_centers_dtype = [
    ('voidID', np.int32), ('center_x', np.float32), ('center_y', np.float32), ('center_z', np.float32),
    ('radius', np.float32), ('void_avg_vx', np.float32), ('void_avg_vy', np.float32), ('void_avg_vz', np.float32)
]
void_centers_array = np.array(void_centers_list, dtype=void_centers_dtype)

# Define normalized radial bins (r / r_v from 0 to 3)
num_bins = 35
s_bins = np.linspace(0, 3, num_bins + 1)
s_bin_centers = 0.5 * (s_bins[:-1] + s_bins[1:])

# Batch size for processing particles
batch_size = 10000

# Loop over each radius bin
for r_min, r_max in bin_ranges:
    print(f"Processing voids with radius {r_min:.1f} to {r_max:.1f} h^-1 Mpc")
    
    # Select voids in the current radius bin
    void_mask = (void_centers_array['radius'] >= r_min) & (void_centers_array['radius'] < r_max)
    selected_voids = void_centers_array[void_mask]
    void_count = len(selected_voids)
    print(f"Number of voids: {void_count}")
    
    if void_count == 0:
        print(f"No voids in radius bin {r_min:.1f} to {r_max:.1f} h^-1 Mpc")
        continue
    
    # Initialize list for profiles
    profiles = []
    
    # Process each void in the radius bin
    for void in selected_voids:
        # Box size
        L = 1000.0  # in h^-1 Mpc
        voidID = void['voidID']
        r_v = void['radius']
        center = np.array([void['center_x'], void['center_y'], void['center_z']], dtype=np.float32)
        v_bulk = np.array([void['void_avg_vx'], void['void_avg_vy'], void['void_avg_vz']], dtype=np.float32)
        
        # Initialize accumulators for this void
        numerator = np.zeros(num_bins, dtype=np.float64)
        denominator = np.zeros(num_bins, dtype=np.float64)
        
        # Process particles in batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch_pos = np.vstack((batch_data['x'], batch_data['y'], batch_data['z'])).T.astype(np.float32)
            
            # Compute differences with minimum image convention
            diff = batch_pos - center
            diff = diff - L * np.round(diff / L)  # Adjust for periodicity
        
            # Compute distances to void center
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
            
            # Select particles within 3*r_v
            mask = distances <= 3 * r_v
            if not np.any(mask):
                continue
            
            # Compute normalized radial distance s = r / r_v
            s = distances[mask] / r_v
            
            # Compute relative velocities
            batch_vel = np.vstack((batch_data['vx'][mask] - v_bulk[0],
                                   batch_data['vy'][mask] - v_bulk[1],
                                   batch_data['vz'][mask] - v_bulk[2])).T.astype(np.float32)
            
            # Radial unit vectors
            #rel_pos = batch_pos[mask] - center
            # Relative positions (already adjusted for PBC)
            rel_pos = diff[mask]
            r_norm = distances[mask]
            r_hat = rel_pos / r_norm[:, np.newaxis]
            
            # Radial velocity component
            v_radial = np.sum(batch_vel * r_hat, axis=1)
            
            # Bin particles
            bin_indices = np.digitize(s, s_bins) - 1
            volumes = batch_data['par_volume'][mask].astype(np.float32)
            for j in range(len(s)):
                bin_idx = bin_indices[j]
                if 0 <= bin_idx < num_bins:
                    numerator[bin_idx] += v_radial[j] * volumes[j]
                    denominator[bin_idx] += volumes[j]
        
        # Compute profile for this void with NaN for no data
        profile = np.full(num_bins, np.nan)
        valid_bins = denominator > 0
        profile[valid_bins] = numerator[valid_bins] / denominator[valid_bins]
        profiles.append(profile)
    
    # Stack profiles and compute statistics
    profiles = np.array(profiles)
    stacked_profile = np.nanmean(profiles, axis=0)
    N = np.sum(~np.isnan(profiles), axis=0)
    std_error = np.nanstd(profiles, axis=0) / np.sqrt(N)
    
    # Compute relative error (set to NaN where velocity is small)
    relative_error = np.where(np.abs(stacked_profile) > 1e-6, std_error / np.abs(stacked_profile), np.nan)
    
    # Save to text file
    output_file = f"{base_path.format(case, redshift, formatted_redshift)}/velocity_profile_r{r_min:.1f}_to_{r_max:.1f}.txt"
    output_data = np.column_stack((s_bin_centers, stacked_profile, std_error, relative_error, N))
    header = (f"# Void radius range: {r_min:.1f} to {r_max:.1f} h^-1 Mpc\n"
              f"# Number of voids: {void_count}\n"
              "# Columns: r/r_v, v_v(r) (km/s), std_error (km/s), relative_error, N")
    np.savetxt(output_file, output_data, delimiter='\t', header=header, fmt='%.6f')
    print(f"Saved velocity profile data to {output_file}")
