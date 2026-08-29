import os
import numpy as np
import cv2
from PIL import Image
try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def create_geotiff_or_png(filepath, img_array, crs="EPSG:4326", transform=None):
    """
    Saves image array as GeoTIFF if rasterio is available, and also saves PNG counterpart.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save PNG version
    png_path = os.path.splitext(filepath)[0] + ".png"
    Image.fromarray(img_array).save(png_path)
    
    # Save GeoTIFF version
    tif_path = os.path.splitext(filepath)[0] + ".tif"
    if HAS_RASTERIO:
        h, w, c = img_array.shape
        if transform is None:
            transform = from_origin(77.5946, 12.9716, 0.0001, 0.0001) # Bangalore area mock geo coords
        
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=c,
            dtype=img_array.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            for band in range(c):
                dst.write(img_array[:, :, band], band + 1)
        print(f"Generated GeoTIFF: {tif_path} & PNG: {png_path}")
    else:
        Image.fromarray(img_array).save(tif_path)
        print(f"Generated PNG/TIFF: {png_path}")

def generate_all_samples():
    output_dir = "dataset/sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = 512, 512
    
    # 1. Single Optical Image (Urban, Forest, Water Body)
    opt_img = np.zeros((h, w, 3), dtype=np.uint8)
    opt_img[:] = [46, 125, 50] # Forest Green background
    # Water river channel curving through middle
    for y in range(h):
        center_x = int(w * 0.45 + 30 * np.sin(y / 50.0))
        opt_img[y, max(0, center_x-35):min(w, center_x+35)] = [30, 136, 229] # Water Blue
    # Built-up urban zone top left
    cv2.rectangle(opt_img, (30, 30), (200, 180), (189, 189, 189), -1) # Concrete grey
    for r in range(40, 170, 30):
        for c in range(40, 190, 30):
            cv2.rectangle(opt_img, (c, r), (c+18, r+18), (120, 144, 156), -1)
    
    create_geotiff_or_png(os.path.join(output_dir, "single_optical.tif"), opt_img)
    
    # 2. Single SAR Image (Grayscale radar backscatter)
    sar_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Background speckle terrain
    noise = np.random.normal(90, 25, (h, w)).clip(0, 255).astype(np.uint8)
    sar_img[:, :, 0] = noise
    sar_img[:, :, 1] = noise
    sar_img[:, :, 2] = noise
    # Water channel = low backscatter (dark specular reflection)
    for y in range(h):
        center_x = int(w * 0.45 + 30 * np.sin(y / 50.0))
        sar_img[y, max(0, center_x-35):min(w, center_x+35)] = [15, 15, 15]
    # Built-up zone = high backscatter (bright double bounce)
    cv2.rectangle(sar_img, (30, 30), (200, 180), (240, 240, 240), -1)
    
    create_geotiff_or_png(os.path.join(output_dir, "single_sar.tif"), sar_img)

    # 3. Bi-Temporal Pair (T1 & T2 showing urban expansion)
    t1_img = opt_img.copy()
    create_geotiff_or_png(os.path.join(output_dir, "bitemporal_t1.tif"), t1_img)
    
    t2_img = t1_img.copy()
    # Urban expansion in South-East quadrant in T2
    cv2.rectangle(t2_img, (300, 300), (480, 480), (207, 216, 220), -1)
    for r in range(310, 470, 35):
        for c in range(310, 470, 35):
            cv2.rectangle(t2_img, (c, r), (c+22, r+22), (90, 107, 124), -1)
            # Add bright rooftop highlights
            cv2.rectangle(t2_img, (c+4, r+4), (c+12, r+12), (255, 112, 67), -1)
    create_geotiff_or_png(os.path.join(output_dir, "bitemporal_t2.tif"), t2_img)

    # 4. Cross-Modal Optical + SAR Pair (Optical has cloud cover, SAR penetrates cloud)
    cross_opt = opt_img.copy()
    # Add simulated cloud cover over urban zone
    cloud = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(cloud, (120, 120), 100, (255, 255, 255), -1)
    cv2.circle(cloud, (180, 100), 80, (250, 250, 250), -1)
    cloud = cv2.GaussianBlur(cloud, (31, 31), 0)
    cross_opt = cv2.addWeighted(cross_opt, 0.5, cloud, 0.5, 0)
    
    create_geotiff_or_png(os.path.join(output_dir, "crossmodal_optical.tif"), cross_opt)
    create_geotiff_or_png(os.path.join(output_dir, "crossmodal_sar.tif"), sar_img)
    print("All sample dataset files generated successfully!")

if __name__ == "__main__":
    generate_all_samples()
