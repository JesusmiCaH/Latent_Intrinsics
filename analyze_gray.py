import numpy as np
from PIL import Image

def analyze_grid_cells(path):
    print(f"Analyzing grid cells of {path}")
    try:
        img = Image.open(path)
        img_np = np.array(img).astype(float)
        
        grid_size = 4
        cell_size = 224
        spacer = 20
        
        block_names = ['img1 (Source)', 'img2 (Target)', 'img3 (Ref)', 'relight_img2', 'relight_img_0_out']
        
        start_x = 0
        for b in range(5):
            print(f"--- Block {b} [{block_names[b]}] ---")
            for r in range(grid_size):
                for c in range(grid_size):
                    y = r * cell_size
                    x = start_x + c * cell_size
                    
                    cell = img_np[y:y+cell_size, x:x+cell_size, :]
                    
                    # Check std
                    std = cell.std()
                    # Check saturation
                    rg_diff = np.abs(cell[:,:,0] - cell[:,:,1]).mean()
                    gb_diff = np.abs(cell[:,:,1] - cell[:,:,2]).mean()
                    rb_diff = np.abs(cell[:,:,0] - cell[:,:,2]).mean()
                    sat = (rg_diff + gb_diff + rb_diff) / 3.0
                    
                    if std < 5 or sat < 2:
                        print(f"  Cell ({r},{c}): GRAY/FLAT detected! Std={std:.2f}, Sat={sat:.2f}")
                    # else:
                    #     print(f"  Cell ({r},{c}): OK. Std={std:.2f}, Sat={sat:.2f}")
            
            start_x += (grid_size * cell_size) + spacer

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_grid_cells("relight_result/relight_100_14.png")
