import numpy as np
import os

import numpy as np
import os
import scipy.ndimage as ndimage

def simulate_occlusions(grid):
    size = grid.shape[0]
    center = size // 2
    # Create polar coordinates from center
    y, x = np.indices((size, size))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    theta = np.arctan2(y - center, x - center)
    
    # Sort pixels by radius to simulate ray progression
    flat_theta = theta.flatten()
    flat_r = r.flatten()
    flat_grid = grid.flatten()
    
    # Group by angular bins (simulating 360 rays)
    num_rays = 360
    theta_bins = (flat_theta + np.pi) / (2 * np.pi) * num_rays
    theta_bins = theta_bins.astype(int) % num_rays
    
    new_grid = np.zeros_like(grid)
    for i in range(num_rays):
        mask = theta_bins == i
        if not np.any(mask): continue
        
        ray_r = flat_r[mask]
        ray_indices = np.where(mask)[0]
        
        # Sort indices by radius for this specific ray
        sort_idx = np.argsort(ray_r)
        sorted_indices = ray_indices[sort_idx]
        
        # Find first hit
        hit = False
        for idx in sorted_indices:
            if not hit:
                if flat_grid[idx] > 0.4: # Threshold for a "hit"
                    hit = True
                    new_grid.flat[idx] = flat_grid[idx]
                else:
                    new_grid.flat[idx] = flat_grid[idx]
            else:
                # Occluded!
                new_grid.flat[idx] = 0.0
    return new_grid

def add_realism(grid):
    # 1. Simulate LiDAR Occlusions (Shadows)
    grid = simulate_occlusions(grid)
    
    # 2. Random Rotation
    angle = np.random.uniform(-30, 30)
    grid = ndimage.rotate(grid, angle, reshape=False, order=1, mode='constant', cval=0)
    
    # 3. Radial Signal Decay (Further = Fainter)
    size = grid.shape[0]
    center = size // 2
    y, x = np.indices((size, size))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    decay = np.clip(1.0 - (r / (size * 0.8)), 0.3, 1.0)
    grid *= decay

    # 4. Clutter (Random small obstacles)
    num_clutter = np.random.randint(0, 4)
    for _ in range(num_clutter):
        cx, cy = np.random.randint(5, size-5, 2)
        grid[cy-1:cy+1, cx-1:cx+1] = np.random.uniform(0.1, 0.5)

    # 5. Gaussian Blur and Noise
    grid = ndimage.gaussian_filter(grid, sigma=np.random.uniform(0.2, 1.0))
    noise = np.random.normal(0, 0.02, grid.shape)
    grid = np.clip(grid + noise, 0, 1)

    return grid

def generate_open_space(size=64):
    grid = np.zeros((size, size), dtype=np.float32)
    # Faint, broken walls at distant edges
    grid[0:2, :] = np.random.uniform(0.1, 0.4)
    grid[-2:, :] = np.random.uniform(0.1, 0.4)
    return add_realism(grid)

def generate_caution_zone(size=64):
    grid = np.zeros((size, size), dtype=np.float32)
    num_clusters = np.random.randint(4, 10)
    for _ in range(num_clusters):
        cx, cy = np.random.randint(10, size-10, 2)
        radius = np.random.randint(2, 5)
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        mask = x*x + y*y <= radius*radius
        grid[mask] = np.random.uniform(0.4, 0.8)
    return add_realism(grid)

def generate_narrow_corridor(size=64):
    grid = np.zeros((size, size), dtype=np.float32)
    gap = np.random.randint(12, 18)
    center = size // 2
    # Vertical walls with some randomness in thickness
    t1 = np.random.randint(2, 5)
    t2 = np.random.randint(2, 5)
    grid[:, center - gap//2 - t1 : center - gap//2] = np.random.uniform(0.6, 1.0)
    grid[:, center + gap//2 : center + gap//2 + t2] = np.random.uniform(0.6, 1.0)
    return add_realism(grid)

def generate_dynamic_obstacle(size=64):
    grid = np.zeros((size, size), dtype=np.float32)
    num_blobs = np.random.randint(1, 3)
    for _ in range(num_blobs):
        cx = size // 2 + np.random.randint(-5, 5)
        cy = size // 2 + np.random.randint(-5, 5)
        radius = np.random.randint(3, 5)
        y, x = np.ogrid[-cy:size-cy, -cx:size-cx]
        mask = x*x + y*y <= radius*radius
        grid[mask] = np.random.uniform(0.7, 1.0)
    return add_realism(grid)

def main():
    np.random.seed(42)
    classes = ['Open Space', 'Caution Zone', 'Narrow Corridor', 'Dynamic Obstacle']
    splits = {'train': 400, 'val': 80, 'test': 80}
    
    data_dir = 'scene_classifier/data'
    os.makedirs(data_dir, exist_ok=True)

    for split_name, count_per_class in splits.items():
        all_data = []
        all_labels = []
        
        for idx, cls_name in enumerate(classes):
            print(f"Generating {count_per_class} samples for {cls_name} ({split_name})...")
            for _ in range(count_per_class):
                if cls_name == 'Open Space':
                    sample = generate_open_space()
                elif cls_name == 'Caution Zone':
                    sample = generate_caution_zone()
                elif cls_name == 'Narrow Corridor':
                    sample = generate_narrow_corridor()
                else: # Dynamic Obstacle
                    sample = generate_dynamic_obstacle()
                
                all_data.append(sample)
                all_labels.append(idx)
        
        all_data = np.array(all_data, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.int64)
        
        # Shuffle
        p = np.random.permutation(len(all_labels))
        all_data = all_data[p]
        all_labels = all_labels[p]
        
        np.save(f'{data_dir}/{split_name}_data.npy', all_data)
        np.save(f'{data_dir}/{split_name}_labels.npy', all_labels)
        print(f"Saved {split_name} split.")

if __name__ == '__main__':
    main()
