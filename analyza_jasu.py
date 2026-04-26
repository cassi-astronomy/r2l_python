import os
import json
import traceback
import rawpy
import exifread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import map_coordinates, gaussian_filter

def get_exif_all(filepath):
    with open(filepath, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    model = str(tags.get('Image Model', 'Unknown')).strip()
    t_tag = tags.get('EXIF ExposureTime') or tags.get('Image ExposureTime')
    iso_tag = tags.get('EXIF ISOSpeedRatings') or tags.get('Image ISOSpeedRatings')
    date_tag = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    t = float(t_tag.values[0].num) / float(t_tag.values[0].den) if t_tag and hasattr(t_tag.values[0], 'num') else 180.0
    iso = float(iso_tag.values[0]) if iso_tag else 400.0
    if not t_tag:
        print(f"  VAROVÁNÍ: {os.path.basename(filepath)} nemá EXIF ExposureTime, používám default 180 s.")
    if not iso_tag:
        print(f"  VAROVÁNÍ: {os.path.basename(filepath)} nemá EXIF ISO, používám default ISO 400.")
    if date_tag and " " in str(date_tag):
        date_part, time_part = str(date_tag).split(" ", 1)
        time_str = f"{date_part.replace(':', '.')} {time_part[:5]}"
    elif date_tag:
        time_str = str(date_tag)
    else:
        time_str = "Neznámý čas"
    return model, t, iso, time_str

def fisheye_to_equirectangular_clean(data, center, radius, tilt_corr=0, out_shape=(1200, 3600)):
    out_h, out_w = out_shape
    y_pano, x_pano = np.indices(out_shape)
    theta = (x_pano / out_w) * 2 * np.pi
    r_base = (y_pano / out_h) * (radius * 1.1)
    r_corrected = r_base + (tilt_corr * np.cos(theta))
    src_x = center[0] + r_corrected * np.cos(theta)
    src_y = center[1] + r_corrected * np.sin(theta)
    with np.errstate(invalid='ignore'):
        panorama = map_coordinates(data, [src_y, src_x], order=1, mode='constant', cval=np.nan)
    return panorama

def get_cmaps():
    # 1. NPS Magnitudy - TVOJE BARVY (od 17.5 do 22.0)
    # Seřazeno od nejsvětlejší (17.5) po nejtmavší (22.0)
    pascal_hex_list = [
        '#ffffff', '#eafff0', '#e8fefc', '#e4ebfb', '#e9e1f9', '#d7bdd6', 
        '#e19bd7', '#e27dcd', '#dc60d2', '#d700d7', '#e600ba', '#e6007f', 
        '#dc0155', '#e0003f', '#e00000', '#e63201', '#e75001', '#e66500', 
        '#eb7801', '#e78601', '#d79b00', '#d2b801', '#dccd00', '#d8d701', 
        '#d2e100', '#bacd01', '#9ccd00', '#60cd00', '#01cd14', '#1ed28b', 
        '#00c8a1', '#00bebe', '#00a1b5', '#0087b4', '#0078b4', '#0064af', 
        '#1a4bb1', '#1923a9', '#0f0091', '#29008c', '#4b017e', '#690069', 
        '#46003c', '#3c0019', '#3c0000', '#1e0000'
    ]
    
    # Vytvoření ListedColormap z tvého seznamu
    nps_cmap = ListedColormap(pascal_hex_list)
    nps_cmap.set_over('#000000') # Nad 22.0 černá
    nps_cmap.set_under('#FFFFFF') # Pod 17.5 bílá
    
    # Hranice pro 46 barev (47 dělících čar)
    nps_norm = BoundaryNorm(np.linspace(17.5, 22.0, len(pascal_hex_list) + 1), nps_cmap.N)
    
    # 2. LUM Kandely (8 dekád)
    decade_bases = [
        ('#4B0082', '#8A2BE2'), ('#00008B', '#0000FF'), ('#008B8B', '#00FFFF'),
        ('#006400', '#228B22'), ('#32CD32', '#ADFF2F'), ('#CCCC00', '#FFFF00'),
        ('#FF8C00', '#FFA500'), ('#8B0000', '#FF0000')
    ]
    full_colors_lum = []
    for b, e in decade_bases:
        seg_cmap = plt.cm.colors.LinearSegmentedColormap.from_list("s", [b, e], N=5)
        for j in range(5): full_colors_lum.append(seg_cmap(j))
    
    lum_cmap = ListedColormap(full_colors_lum)
    lum_cmap.set_bad('#000000')
    lum_norm = BoundaryNorm(np.logspace(-4, 4, 41), lum_cmap.N)
    
    return nps_cmap, nps_norm, lum_cmap, lum_norm

def process_images():
    with open('config.json', 'r') as f:
        full_cfg = json.load(f)

    for d in ['vstup_sky', 'vstup_dark', 'vystup']:
        if not os.path.exists(d): os.makedirs(d)
    
    master_dark = None
    dark_files = [f for f in os.listdir('vstup_dark') if f.lower().endswith('.cr2')]
    if dark_files:
        darks = []
        for dark_file in dark_files:
            dark_path = os.path.join('vstup_dark', dark_file)
            with rawpy.imread(dark_path) as dark_raw:
                darks.append(dark_raw.raw_image.astype(np.float32))
        master_dark = np.mean(darks, axis=0)

    sky_files = [f for f in os.listdir('vstup_sky') if f.lower().endswith('.cr2')]
    n_cmap, n_norm, l_cmap, l_norm = get_cmaps()
    for sf in sky_files:
        path = os.path.join('vstup_sky', sf)
        model, t, iso, f_date = get_exif_all(path)
        if model not in full_cfg:
            print(f"Přeskakuji {sf}: model '{model}' není v config.json.")
            continue
        if t <= 0 or iso <= 0:
            print(f"Přeskakuji {sf}: neplatné EXIF hodnoty t={t}, ISO={iso}.")
            continue
        cfg = full_cfg[model]
        print(f"Zpracovávám: {sf} | Tilt: {cfg.get('tilt_correction', 0)}")
        base_name, _ = os.path.splitext(sf)
        
        try:
            with rawpy.imread(path) as raw:
                data = raw.raw_image.astype(np.float32)
                if master_dark is not None and data.shape == master_dark.shape:
                    data = np.clip(data - master_dark, 1.0, None)

                # Velmi jemný filtr pro potlačení digitálního šumu mřížky
                data_subtle = gaussian_filter(data, sigma=0.5)

                c_x, c_y = cfg['circle_center']; rad = cfg['circle_radius']
                yy, xx = np.indices(data.shape)
                r_map = np.sqrt((xx - c_x)**2 + (yy - c_y)**2)
                
                vignette = 1.0 + (cfg.get('vignetting_coeff', 0) / 100.0) * ((r_map / rad)**2)
                adu_norm = (data_subtle * vignette) / (t * (iso / 400.0))
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    mag_data = -2.5 * np.log10(adu_norm + 1e-9) + cfg['calibration_constant']
                cd_data = 10.8e4 * 10**(-0.4 * mag_data)
                
                zenit_val = np.nanmedian(mag_data[r_map < (rad * 0.2)])
                label_base = f"Zenit: {zenit_val:.2f} MSA | {f_date} | {model}"

                # --- 1. NPS Kruh ---
                fig, ax = plt.subplots(figsize=(10, 11), facecolor='black')
                m_masked = np.copy(mag_data); m_masked[r_map > rad] = 99
                crop1 = m_masked[int(c_y-rad):int(c_y+rad), int(c_x-rad):int(c_x+rad)]
                img1 = ax.imshow(crop1, cmap=n_cmap, norm=n_norm)
                ax.set_axis_off()
                cax1 = make_axes_locatable(ax).append_axes("bottom", size="3%", pad=0.1)
                fig.colorbar(img1, cax=cax1, orientation='horizontal', ticks=np.arange(17.5, 23.0, 0.5))
                cax1.xaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)
                cax1.set_xlabel(f"Jas oblohy [mag/arcsec²] | {label_base}", color='white')
                plt.savefig(os.path.join('vystup', f'{base_name}_NPS.jpg'), bbox_inches='tight', facecolor='black', dpi=300)
                plt.close()

                # --- 2. LUM Kruh ---
                fig, ax = plt.subplots(figsize=(10, 11), facecolor='black')
                cd_masked = np.copy(cd_data); cd_masked[r_map > rad] = np.nan
                crop2 = cd_masked[int(c_y-rad):int(c_y+rad), int(c_x-rad):int(c_x+rad)]
                img2 = ax.imshow(crop2, cmap=l_cmap, norm=l_norm)
                ax.set_axis_off()
                cax2 = make_axes_locatable(ax).append_axes("bottom", size="3%", pad=0.1)
                l_ticks = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
                l_labels = ['0.0001', '', '0.001', '', '0.01', '', '0.1', '0.5', '1', '5', '10', '50', '100', '500', '1000', '5k', '10k']
                cbar2 = fig.colorbar(img2, cax=cax2, orientation='horizontal', ticks=l_ticks)
                cax2.set_xticklabels(l_labels, fontsize=7)
                cax2.xaxis.set_tick_params(color='white', labelcolor='white')
                cax2.set_xlabel(f"Jas oblohy [cd/m²] | {label_base}", color='white')
                plt.savefig(os.path.join('vystup', f'{base_name}_lum.jpg'), bbox_inches='tight', facecolor='black', dpi=300)
                plt.close()

                # --- 3. PANO ---
                mag_pano_data = gaussian_filter(mag_data, sigma=0.5)
                pano = fisheye_to_equirectangular_clean(mag_pano_data, (c_x, c_y), rad, tilt_corr=cfg.get('tilt_correction', 0))
                pano[np.isnan(pano)] = 99
                fig, ax = plt.subplots(figsize=(18, 10), facecolor='black')
                img3 = ax.imshow(pano, cmap=n_cmap, norm=n_norm, aspect='auto')
                ax.set_axis_off()
                cax3 = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.5)
                fig.colorbar(img3, cax=cax3, orientation='horizontal', ticks=np.arange(17.5, 23.0, 0.5))
                cax3.xaxis.set_tick_params(color='white', labelcolor='white', labelsize=10)
                cax3.set_xlabel("Jas oblohy [mag/arcsec²]", color='white')
                plt.title(label_base, color='white', pad=20, fontsize=12)
                plt.savefig(os.path.join('vystup', f'{base_name}_pano.jpg'), bbox_inches='tight', facecolor='black', dpi=300)
                plt.close()

                print(f"  Hotovo.")
        except Exception as e:
            print(f"  CHYBA: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    process_images()
