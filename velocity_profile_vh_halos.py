
import numpy as np
import sys

# --------------------------- CLI & Paths ---------------------------
if len(sys.argv) < 3:
    print("Usage: python compute_void_velocity_dispersion_individual.py <case> <redshift>")
    sys.exit(1)

case = sys.argv[1]
redshift = sys.argv[2]
formatted_redshift = "{:.2f}".format(float(redshift))
type_ss = "Halos"


base_path = "/project/MCChu/khluo/Downloads/vide_public/void_vide/Cobaya_DESI/1024/{}/{}/z_{}/snapshot_1024/output/sim_halos_minnone/sample_sim_halos_minnone_z{}_d00"
input_file = f"{base_path.format(case, type_ss, redshift, formatted_redshift)}/match_all_particles.npy"

# --------------------------- Configuration ---------------------------
# Void radius bins (h^-1 Mpc) — add more as needed
bin_ranges = [
    (7.0, 12.0),
    (12.0, 17.0),
    (17.0, 22.0)
]

# Normalized radius s=r/rv
num_bins = 35
s_edges = np.linspace(0.0, 3.0, num_bins + 1)
s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

# Periodic box size (h^-1 Mpc)
L = 1000.0

# --------------------------- Load data ---------------------------
print(f"Loading: {input_file}")
data = np.load(input_file)

required_fields = [
    'voidID', 'center_x', 'center_y', 'center_z', 'radius',
    'void_avg_vx', 'void_avg_vy', 'void_avg_vz',
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'par_volume'
]
missing = [f for f in required_fields if f not in data.dtype.names]
if missing:
    raise RuntimeError(f"Input file missing required fields: {missing}")

N = len(data)
print(f"Total rows (void-particle pairs): {N}")

# --------------------------- Vector views ---------------------------
voidID_all = data['voidID']
rv_all  = data['radius'].astype(np.float32)

cx_all = data['center_x'].astype(np.float32)
cy_all = data['center_y'].astype(np.float32)
cz_all = data['center_z'].astype(np.float32)

vbx_all = data['void_avg_vx'].astype(np.float32)
vby_all = data['void_avg_vy'].astype(np.float32)
vbz_all = data['void_avg_vz'].astype(np.float32)


x_all  = data['x'].astype(np.float32)
y_all  = data['y'].astype(np.float32)
z_all  = data['z'].astype(np.float32)

vx_all = data['vx'].astype(np.float32)
vy_all = data['vy'].astype(np.float32)
vz_all = data['vz'].astype(np.float32)

w_all  = data['par_volume'].astype(np.float64)

# --------------------------- Group by voidID ---------------------------
# Sort by voidID to make contiguous groups
sort_idx = np.argsort(voidID_all)
voidID_sorted = voidID_all[sort_idx]
unique_voids, start_idx = np.unique(voidID_sorted, return_index=True)
end_idx = np.empty_like(start_idx)
end_idx[:-1] = start_idx[1:]
end_idx[-1] = len(sort_idx)

print(f"Unique voids: {len(unique_voids)}")

# --------------------------- Main loop over radius bins ---------------------------
for r_min, r_max in bin_ranges:
    print(f"\n=== Individual stacking for void radius {r_min:.1f}–{r_max:.1f} h^-1 Mpc ===")

    # Select void groups within the radius bin (take r_v from first row in the group)
    selected_groups = []
    for v, s0, s1 in zip(unique_voids, start_idx, end_idx):
        i0 = sort_idx[s0]
        r_v = rv_all[i0]
        if (r_v >= r_min) and (r_v < r_max):
            selected_groups.append((v, s0, s1))

    void_count = len(selected_groups)
    print(f"Number of voids in selection: {void_count}")
    if void_count == 0:
        print("No voids; skipping.")
        continue

    # Hold per-void profiles
    per_void_mean  = []  # u_v^{(j)}(s)
    per_void_sigma = []  # σ_v^{(j)}(s) via direct deviations about u_v^{(j)}(s)

    # ------------------ Per-void computations ------------------
    for idx_void, (vID, s0, s1) in enumerate(selected_groups, start=1):
        rows = sort_idx[s0:s1]
        i0   = rows[0]

        r_v = float(rv_all[i0])
        center = np.array([cx_all[i0], cy_all[i0], cz_all[i0]], dtype=np.float32)
        v_bulk = np.array([0, 0, 0], dtype=np.float32)
        #v_bulk = np.array([vbx_all[i0], vby_all[i0], vbz_all[i0]], dtype=np.float32)

        # Positions relative to center with periodic BC
        pos  = np.column_stack((x_all[rows], y_all[rows], z_all[rows]))  # (m,3)
        diff = pos - center
        diff = diff - L * np.round(diff / L)

        r = np.sqrt(np.sum(diff**2, axis=1)).astype(np.float32)
        s = (r / r_v).astype(np.float64)

        # Keep within 3 r_v and s in [0, 3]
        m = (s >= s_edges[0]) & (s <= s_edges[-1]) & (r <= 3.0 * r_v)

        # Initialize per-void profiles
        mean_profile  = np.full(num_bins, np.nan, dtype=np.float64)
        sigma_profile = np.full(num_bins, np.nan, dtype=np.float64)

        if np.any(m):
            diff_m = diff[m]
            r_m    = r[m].astype(np.float32)
            s_m    = s[m].astype(np.float64)
            r_hat  = diff_m / r_m[:, None]  # unit radial vectors

            # Velocities relative to void bulk; project radially
            vel  = np.column_stack((vx_all[rows][m], vy_all[rows][m], vz_all[rows][m])).astype(np.float32)
            v_rel = vel - v_bulk
            v_rad = np.sum(v_rel * r_hat, axis=1).astype(np.float64)

            # Volume weights
            w = w_all[rows][m].astype(np.float64)

            # Bin indices for s
            b = np.digitize(s_m, s_edges) - 1
            inrange = (b >= 0) & (b < num_bins)

            if np.any(inrange):
                b    = b[inrange]
                w_b  = w[inrange]
                vr_b = v_rad[inrange]

                # ----- Per-void mean: u_v^{(j)}(s) -----
                sum_w   = np.bincount(b, weights=w_b,        minlength=num_bins)
                sum_w_v = np.bincount(b, weights=w_b * vr_b, minlength=num_bins)

                valid_mean_bins = (sum_w > 0.0)
                mean_profile[valid_mean_bins] = sum_w_v[valid_mean_bins] / sum_w[valid_mean_bins]

                # ----- Per-void dispersion: σ_v^{(j)}(s) via direct deviations about u_v^{(j)}(s) -----
                # Only use samples whose bin has a defined mean
                sample_valid = valid_mean_bins[b]
                if np.any(sample_valid):
                    dev = (vr_b[sample_valid] - mean_profile[b[sample_valid]]).astype(np.float64)
                    sum_w_dev2 = np.bincount(
                        b[sample_valid],
                        weights=w_b[sample_valid] * dev**2,
                        minlength=num_bins
                    )

                    use = valid_mean_bins & (sum_w_dev2 > 0.0)
                    # Numerical safety against tiny negative due to FP roundoff
                    var = np.zeros(num_bins, dtype=np.float64)
                    var[use] = sum_w_dev2[use] / sum_w[use]
                    var[var < 0.0] = 0.0
                    sigma_profile[use] = np.sqrt(var[use])

        # Store per-void profiles
        per_void_mean.append(mean_profile)
        per_void_sigma.append(sigma_profile)

        if (idx_void % 50) == 0 or idx_void == void_count:
            print(f"  processed void {idx_void}/{void_count}", end='\r', flush=True)

    # ------------------ Stack across voids (equal weight per void) ------------------
    per_void_mean  = np.array(per_void_mean)   # (N_voids, num_bins)
    per_void_sigma = np.array(per_void_sigma)  # (N_voids, num_bins)

    # Stacked mean <v_rad>(s)
    stacked_mean = np.nanmean(per_void_mean, axis=0)
    N_mu         = np.sum(~np.isnan(per_void_mean), axis=0)
    std_error_mu = np.nanstd(per_void_mean, axis=0) / np.sqrt(np.maximum(N_mu, 1))
    rel_error_mu = np.where(np.abs(stacked_mean) > 1e-6, std_error_mu / np.abs(stacked_mean), np.nan)

    # Stacked dispersion σ_v(s)
    stacked_sigma   = np.nanmean(per_void_sigma, axis=0)
    N_sigma         = np.sum(~np.isnan(per_void_sigma), axis=0)
    std_error_sigma = np.nanstd(per_void_sigma, axis=0) / np.sqrt(np.maximum(N_sigma, 1))
    rel_error_sigma = np.where(np.abs(stacked_sigma) > 1e-6, std_error_sigma / np.abs(stacked_sigma), np.nan)

    # ------------------ Save combined output ------------------
    out_dir = base_path.format(case, type_ss, redshift, formatted_redshift)
    out_file = f"{out_dir}/velocity_profile_void_r{r_min:.1f}_to_{r_max:.1f}_vh_individual.txt"
    combined = np.column_stack((
        s_centers,          # 1: s=r/rv
        stacked_mean,       # 2: <v_rad>(km/s)
        std_error_mu,       # 3: std_error_v(km/s)
        rel_error_mu,       # 4: rel_error_v
        N_mu,               # 5: N_voids_for_mean
        stacked_sigma,      # 6: sigma_v(km/s)
        std_error_sigma,    # 7: std_error_sigma(km/s)
        rel_error_sigma,    # 8: rel_error_sigma
        N_sigma             # 9: N_voids_for_sigma
    ))

    header = (
        f"# Individual-stacked void radial velocity & dispersion (volume-weighted)\n"
        f"# Radius selection: r_v in [{r_min:.1f}, {r_max:.1f}] h^-1 Mpc\n"
        f"# Per-void mean:   u_v^j(s) = (Σ w v_rad)/(Σ w)\n"
        f"# Per-void disp.:  σ_v^j(s) = sqrt( Σ w [v_rad - u_v^j(s)]^2 / Σ w )  [direct deviations]\n"
        f"# Stacked across voids (equal weight per void): u_V(s) = ⟨u_v^j(s)⟩;  σ_V(s) = ⟨σ_v^j(s)⟩\n"
        f"# s-bins: s in [0,3], num_bins={num_bins}\n"
        f"# Weights: w = par_volume (Voronoi/cell volume)\n"
        f"# Periodic BC: L = {L:.1f} h^-1 Mpc (minimum-image convention)\n"
        f"# Columns:\n"
        f"# 1: s=r/rv\n"
        f"# 2: <v_rad>(km/s) — stacked mean across voids\n"
        f"# 3: std_error_v(km/s)\n"
        f"# 4: rel_error_v\n"
        f"# 5: N_voids_for_mean\n"
        f"# 6: sigma_v(km/s) — stacked dispersion across voids\n"
        f"# 7: std_error_sigma(km/s)\n"
        f"# 8: rel_error_sigma\n"
        f"# 9: N_voids_for_sigma\n"
    )

    np.savetxt(
        out_file,
        combined,
        delimiter='\t',
        header=header,
        fmt=['%.6f','%.6f','%.6f','%.6f','%d','%.6f','%.6f','%.6f','%d']
    )
    print(f"\nSaved individual-stacked velocity+dispersion to:\n  {out_file}")
