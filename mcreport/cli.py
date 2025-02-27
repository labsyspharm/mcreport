#!/usr/bin/env python3
"""
Command‑line version of MCMICRO.report
Misha Ermakov (LSP HiTS 2025)


"""

import argparse
import os
import sys
import re
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as mpatches
import scimap
import anndata as ad
import tifffile
from collections import defaultdict
from joblib import Parallel, delayed
from shapely.geometry import Polygon as ShapelyPolygon, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from fpdf import FPDF
from PIL import Image
from PIL.TiffTags import TAGS
import alphashape
import shapely

# ---------------------------
# Utility Functions
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MCMICRO.report"
    )
    parser.add_argument("--ome_path", type=str,
                        metavar="/ path / to / .ome.tiff",
                        help="Path to the OME TIFF file")
    parser.add_argument("--h5ad_path", type=str,
                        metavar="/ path / to / .h5ad or .csv",
                        help="Path to the .h5ad (or CSV) file")
    parser.add_argument("--HnE_path", type=str,
                        metavar="/ path / to / HnE.tiff",
                        help="Path to the H&E file")
    parser.add_argument("--output_path", type=str,
                        metavar="/ path / to /output / folder",
                        help="Output directory path")
    parser.add_argument("--reference_folder", type=str,
                        metavar="/ reference / files ?? ",
                        help="Path to the reference files folder")
    parser.add_argument("--markers_csv", type=str,
                        metavar="/ Path / to / markers.csv",
                        help="Path to markers CSV")
    parser.add_argument("--phenotypes_csv", type=str,
                        metavar="/path / to // phenotyping table.csv ",
                        help="Path to phenotypes CSV")
    parser.add_argument("--gates", type=str,
                        metavar="/path / to / gates.csv",
                        help="Path to gates CSV")
    parser.add_argument("--alternative_roi_path", type=str,
                        metavar="/path / to / roi.csv",
                        help="Alternative ROI CSV path")
    parser.add_argument("--workflow_type", type=str,
                        choices=["Marker", "HnE", "Human"],
                        default="Marker",
                        help="Workflow type")
    
    parser.add_argument("--sampleid", type=str, default="SAMPLE", help="Sample ID")
    parser.add_argument("--author", type=str, default="USER", help="Author name")
    parser.add_argument("--sample_marker", type=str, default="Hoechst", help="DNA marker")
    parser.add_argument("--tumor_marker", type=str, default="Pan-CK", help="Tumor marker")
    parser.add_argument("--alpha_value_tumor", type=float, default=0.001, help="Alpha value for tumor clustering")
    return parser.parse_args()

def compute_areas(poly, px_size):
    area_px = poly.area
    if px_size is not None:
        area_um2 = area_px * (px_size ** 2)
        area_mm2 = area_um2 / 1e6
    else:
        area_um2 = None
        area_mm2 = None
    return area_px, area_um2, area_mm2

def parse_polygon_coords(all_points_str):
    coord_pairs = all_points_str.strip().split()
    coords = []
    for cp in coord_pairs:
        x_str, y_str = cp.split(',')
        coords.append((float(x_str), float(y_str)))
    return coords

def addROI_omero(adata, roi, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                 imageid='imageid', subset=None, overwrite=True,
                 label='ROI', n_jobs=-1, verbose=False):
    data = pd.DataFrame(adata.obs)[[x_coordinate, y_coordinate, imageid]]
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]
        sub_data = data[data[imageid].isin(subset)]
    else:
        sub_data = data

    def parse_roi_points(all_points):
        return np.array(re.findall(r'\d+\.?\d+', all_points), dtype=float).reshape(-1, 2)

    def ellipse_points_to_patch(vertex_1, vertex_2, co_vertex_1, co_vertex_2):
        v_and_co_v = np.array([vertex_1, vertex_2, co_vertex_1, co_vertex_2])
        centers = v_and_co_v.mean(axis=0)
        d = np.linalg.norm(v_and_co_v - centers, axis=1)
        width = d[0] * 2
        height = d[1] * 2
        vector_2 = v_and_co_v[1] - v_and_co_v[0]
        vector_2 /= np.linalg.norm(vector_2)
        angle = np.degrees(np.arccos(np.dot([1, 0], vector_2)))
        ellipse_patch = mpatches.Ellipse(centers, width=width, height=height, angle=angle)
        return ellipse_patch

    def get_mpatch(roi):
        points = parse_roi_points(roi['all_points'])
        roi_type = roi['type']
        if roi_type in ['Point', 'Line']:
            roi_mpatch = mpatches.Polygon(points, closed=False)
        elif roi_type in ['Rectangle', 'Polygon', 'Polyline']:
            roi_mpatch = mpatches.Polygon(points, closed=True)
        elif roi_type == 'Ellipse':
            roi_mpatch = ellipse_points_to_patch(*points)
        else:
            raise ValueError(f"ROI type {roi_type} not recognized")
        return roi_mpatch

    def add_roi_internal(roi_id):
        roi_subset = roi[roi['Id'] == roi_id].iloc[0]
        roi_mpatch = get_mpatch(roi_subset)
        inside = sub_data[roi_mpatch.contains_points(sub_data[[x_coordinate, y_coordinate]])]
        inside['ROI_internal'] = roi_subset['Name']
        return inside

    roi_list = roi['Id'].unique()
    final_roi = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(add_roi_internal)(roi_id=i) for i in roi_list)
    final_roi = pd.concat(final_roi)[['ROI_internal']]
    result = pd.merge(data, final_roi, left_index=True, right_index=True, how='outer')
    result = result.reindex(adata.obs.index)
    if label in adata.obs.columns:
        if not overwrite:
            old_roi = adata.obs[label]
            combined_roi = pd.merge(result, old_roi, left_index=True, right_index=True, how='outer')
            combined_roi['ROI_internal'] = combined_roi['ROI_internal'].fillna(combined_roi[label])
        else:
            combined_roi = result.copy()
            combined_roi['ROI_internal'] = combined_roi['ROI_internal'].fillna('Other')
    else:
        combined_roi = result.copy()
        combined_roi['ROI_internal'] = combined_roi['ROI_internal'].fillna('Other')
    adata.obs[label] = combined_roi['ROI_internal']
    return adata

def addROI_shapely(adata, band_poly, label='Band_ROI', xcol='X_centroid', ycol='Y_centroid'):
    if isinstance(band_poly, list):
        band_poly = MultiPolygon(band_poly)
    coords = adata.obs[[xcol, ycol]].values  
    inside_mask = []
    for (cx, cy) in coords:
        pt = Point(cx, cy)
        inside_mask.append(band_poly.contains(pt))
    inside_mask = pd.Series(inside_mask, index=adata.obs.index).replace({True: label, False: 'False'})
    adata.obs[label] = inside_mask
    return adata

def build_adjacency(df_tree):
    adjacency = defaultdict(list)
    all_nodes = set()
    for _, row in df_tree.iterrows():
        parent = row.iloc[0]
        child = row.iloc[1]
        adjacency[parent].append(child)
        all_nodes.add(parent)
        all_nodes.add(child)
    return adjacency, all_nodes

def get_all_descendants(node, adjacency):
    stack = [node]
    visited = set()
    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            stack.extend(adjacency[current])
    return visited

def compute_counts_node_and_roi(adata, adjacency, all_nodes, phenotype_col="phenotype", roi_col="Tumor_ROI"):
    phenotype_series = adata.obs[phenotype_col]
    roi_series = adata.obs[roi_col]
    rois = sorted(roi_series.dropna().unique())
    node_to_desc = {}
    for node in all_nodes:
        node_to_desc[node] = get_all_descendants(node, adjacency)
    rows = []
    for node in sorted(all_nodes):
        desc = node_to_desc[node]
        total_count = phenotype_series.isin(desc).sum()
        row_dict = {"Cell type": node, "Total": total_count}
        for roi_label in rois:
            mask = (roi_series == roi_label) & (phenotype_series.isin(desc))
            row_dict[str(roi_label)] = mask.sum()
        rows.append(row_dict)
    return pd.DataFrame(rows)

def get_hierarchical_order(adjacency, root):
    order = []
    def dfs(node):
        order.append(node)
        for child in adjacency[node]:
            dfs(child)
    dfs(root)
    return order

def plot_h_and_e_and_two_markers(h_and_e, image_path, marker_csv, marker_1_name,
                                 marker_2_name, sample_df, tumor_df, output_dir,
                                 pixel_size_x, workflow_type_key, sz=10):
    # Downsample the H&E image
    if h_and_e.ndim == 3 and h_and_e.shape[0] < 10:
        h_and_e = h_and_e.transpose([1, 2, 0])
    h_and_e_small = h_and_e[::sz, ::sz]
    marker_df = pd.read_csv(marker_csv)
    channel_index_1 = marker_df.index[marker_df["marker_name"] == marker_1_name][0]
    ome_data_1 = tifffile.imread(image_path, key=channel_index_1)
    marker_small_1 = ome_data_1[::sz, ::sz]
    height_small, width_small = marker_small_1.shape[:2]
    extent = [0, width_small * pixel_size_x, height_small * pixel_size_x, 0]
    if workflow_type_key == 'Marker':
        channel_index_2 = marker_df.index[marker_df["marker_name"] == marker_2_name][0]
        ome_data_2 = tifffile.imread(image_path, key=channel_index_2)
        marker_small_2 = ome_data_2[::sz, ::sz]
    # Get polygons from CSVs
    sample_polygons = []
    for _, row in sample_df.iterrows():
        coords = parse_polygon_coords(row["all_points"])
        sample_polygons.append(ShapelyPolygon(coords))
    tumor_polygons = []
    for _, row in tumor_df.iterrows():
        coords = parse_polygon_coords(row["all_points"])
        tumor_polygons.append(ShapelyPolygon(coords))
    fig, (ax_hne, ax_markers) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    ax_hne.imshow(h_and_e_small, extent=extent)
    ax_hne.tick_params(axis='both', labelsize=7)
    ax_hne.set_xlabel("mm", fontsize=7)
    ax_hne.set_ylabel("mm", fontsize=7)
    for spine in ["top", "right", "left", "bottom"]:
        ax_hne.spines[spine].set_visible(False)
    sample_color_roi = '#081d58'
    tumor_color_roi = '#cc4c02'
    for poly in sample_polygons:
        coords = (np.array(poly.exterior.coords) / sz) * pixel_size_x
        ax_hne.plot(coords[:, 0], coords[:, 1], color=sample_color_roi, linewidth=2, linestyle='dotted')
        ax_markers.plot(coords[:, 0], coords[:, 1], color=sample_color_roi, linewidth=2, linestyle='dotted')
    for poly in tumor_polygons:
        coords = (np.array(poly.exterior.coords) / sz) * pixel_size_x
        ax_hne.plot(coords[:, 0], coords[:, 1], color=tumor_color_roi, linewidth=2, linestyle='dotted')
        ax_markers.plot(coords[:, 0], coords[:, 1], color=tumor_color_roi, linewidth=2, linestyle='dotted')
    sample_area_total = sample_df["area_mm2"].sum()
    text_x = 10
    text_y = (h_and_e_small.shape[0] * pixel_size_x) - 200
    if workflow_type_key == 'Marker':
        ax_hne_text = f"Area: {marker_2_name}"
    elif workflow_type_key == 'Human':
        ax_hne_text = "Tumor area"
    else:
        ax_hne_text = "Detected tumor"
    ax_hne.text(x=text_x, y=text_y, s=f"Sample: {sample_area_total:.2f} mm²", ha="left", va="top",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=sample_color_roi, ec="none", alpha=0.8))
    ax_hne.text(x=text_x, y=text_y - 150, s=ax_hne_text, ha="left", va="top",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc=tumor_color_roi, ec="none", alpha=0.8))
    ax_hne.set_title("Detected ROI:", size=8, loc='left')
    if workflow_type_key == 'Marker':
        marker_small_1 = np.max(marker_small_1) - marker_small_1
        cmap_1 = matplotlib.colors.ListedColormap([sample_color_roi, 'white'])
        ax_markers.imshow(np.sqrt(marker_small_1), cmap=cmap_1,
                          vmin=np.percentile(np.sqrt(marker_small_1), 20),
                          vmax=np.percentile(np.sqrt(marker_small_1), 80), extent=extent)
        marker_small_2 = np.max(marker_small_2) - marker_small_2
        cmap_2 = matplotlib.colors.ListedColormap([tumor_color_roi, 'white'])
        ax_markers.imshow(np.sqrt(marker_small_2), cmap=cmap_2,
                          vmin=np.percentile(np.sqrt(marker_small_2), 5),
                          vmax=np.percentile(np.sqrt(marker_small_2), 95), alpha=0.5, extent=extent)
    else:
        marker_small_1 = np.max(marker_small_1) - marker_small_1
        cmap_1 = matplotlib.colors.ListedColormap([sample_color_roi, 'white'])
        ax_markers.imshow(np.sqrt(marker_small_1), cmap=cmap_1,
                          vmin=np.percentile(np.sqrt(marker_small_1), 5),
                          vmax=np.percentile(np.sqrt(marker_small_1), 95), extent=extent)
    ax_markers.set_xlabel("mm", fontsize=7)
    ax_markers.tick_params(axis='both', labelsize=7)
    ax_markers.set_ylabel("")
    ax_markers.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax_markers.spines[spine].set_visible(False)
    if workflow_type_key == 'Marker':
        ax_markers.set_title(f"DNA and expression of {marker_2_name}:", size=8, loc='left')
    else:
        ax_markers.set_title("DNA distribution:", size=8, loc='left')
    outname = "sample_he_plot.png"
    plt.savefig(os.path.join(output_dir, outname), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined figure to {os.path.join(output_dir, outname)}")

# PDF report generation

class MyPDF(FPDF):
    def header(self):
        logo_path = os.path.join(self.reference_folder, 'logo-pathreport-mcmicro.png')
        if os.path.exists(logo_path):
            self.image(logo_path, x=10, y=5, w=140)
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 9)
        self.cell(0, 5, f"{self.author}, {datetime.datetime.now().strftime('%Y-%m-%d')}", 0, 0)
        self.cell(0, 5, "For Research Use Only", 0, 0, align='R')

def create_pdf(pdf, output, sampleid, df_counts, workflow_type, tumor_marker,
               df_cells_mm2, immunoscore_df, author):
    pdf.add_page()
    pdf.image(os.path.join(output, "sample_he_plot.png"), x=8, y=35, w=200)
    pdf.image(os.path.join(output, "cell_ratios_and_counts_plot.png"), x=8, y=115, w=200)
    pdf.set_xy(10, 20)
    pdf.set_font("Arial", size=9, style="B")
    pdf.cell(0, 5, "SampleID:", ln=False)
    pdf.set_xy(27, 20)
    pdf.set_font("Arial", size=9)
    pdf.cell(0, 5, f" {sampleid}", ln=True)
    pdf.set_xy(100, 20)
    pdf.set_font("Arial", size=9, style="B")
    pdf.cell(0, 5, "Total Cells:", ln=False)
    pdf.set_xy(120, 20)
    pdf.set_font("Arial", size=9)
    total_cells = df_counts.loc[df_counts['Cell type'] == 'all', 'Total'].values[0] if 'all' in df_counts['Cell type'].values else 0
    pdf.cell(0, 5, f" {total_cells}", ln=True)
    pdf.set_xy(100, 25)
    pdf.set_font("Arial", size=9, style="B")
    pdf.cell(0, 5, "Phenotyped Cells:", ln=False)
    pdf.set_xy(130, 25)
    pdf.set_font("Arial", size=9)
    if 'all' in df_counts['Cell type'].values and 'Unknown' in df_counts['Cell type'].values:
        pos_counts = df_counts.loc[df_counts['Cell type'] == 'all', 'Total'].iloc[0] - df_counts.loc[df_counts['Cell type'] == 'Unknown', 'Total'].iloc[0]
        pos_percentage = round(pos_counts / df_counts.loc[df_counts['Cell type'] == 'all', 'Total'].iloc[0] * 100)
    else:
        pos_counts, pos_percentage = 0, 0
    pdf.cell(0, 5, f"{pos_counts} ({pos_percentage}%)", ln=True)
    pdf.set_xy(10, 25)
    pdf.set_font("Arial", size=9, style="B")
    pdf.cell(0, 5, "Workflow:", ln=False)
    pdf.set_xy(27, 25)
    pdf.set_font("Arial", size=9)
    if workflow_type == 'Marker-based workflow':
        pdf.cell(0, 5, f" {workflow_type} ({tumor_marker})", ln=True)
    else:
        pdf.cell(0, 5, f"{workflow_type}", ln=True)
    pdf.set_xy(10, 185)
    pdf.set_font("Arial", size=9, style="B")
    headers = ["Cell type", "Total (per mm2)", "Tumor area (per mm2)", "Other (per mm2)"]
    cell_widths = [50, 50, 50, 50]
    for i, header in enumerate(headers):
        pdf.cell(cell_widths[i], 5, header, 0)
    pdf.ln()
    pdf.set_font("Arial", size=9)
    for _, row in df_cells_mm2.iterrows():
        cell_type_text = str(row['Cell type'])
        pdf.cell(50, 4, cell_type_text, 0)
        pdf.cell(50, 4, str(row['Total']), 0)
        pdf.cell(50, 4, str(row['Tumor area']), 0)
        pdf.cell(50, 4, str(row['Other']), 0)
        pdf.ln()
    if not immunoscore_df.empty:
        pdf.add_page()
        pdf.set_xy(10, 25)
        pdf.set_font("Arial", size=11, style="B")
        pdf.cell(0, 5, "Immunoscore:", ln=True)
        pdf.set_xy(10, 30)
        pdf.set_font("Arial", size=9, style="B")
        headers = ["Phenotypes and Scores", "Tumor (per mm2)", "TSI (per mm2)", "Tumor + TSI (per mm2)"]
        cell_widths = [50, 50, 50, 50]
        for i, header in enumerate(headers):
            pdf.cell(cell_widths[i], 5, header, 0)
        pdf.ln()
        pdf.set_font("Arial", size=9)
        for idx, row in immunoscore_df.iterrows():
            cell_type_text = str(idx)
            pdf.cell(50, 5, cell_type_text, 0)
            pdf.cell(50, 5, str(row.get('Tumor', '')), 0)
            pdf.cell(50, 5, str(row.get('TSI', '')), 0)
            pdf.cell(50, 5, str(row.get('Tumor+TSI', '')), 0)
            pdf.ln()
    pdf.output(os.path.join(output, f"report_{workflow_type.split()[0]}.pdf"))
    print(f"PDF created and saved as {os.path.join(output, f'report_{workflow_type.split()[0]}.pdf')}")

# ---------------------------
# Main Function
# ---------------------------

def main():
    args = parse_args()
    print("Running MCMICRO.report command‑line script …")
    start_time = time.time()
    
    # Set parameters from args
    alpha_value_tumor = args.alpha_value_tumor
    immunoProfile = False  # Set to True to run immunoprofile steps
    sampleid = args.sampleid
    author = args.author
    sample_marker = args.sample_marker
    workflow_type_key = args.workflow_type
    tumor_marker = args.tumor_marker

    # Define paths
    ome_path = args.ome_path
    h5ad_path = args.h5ad_path
    HnE_path = args.HnE_path
    output = os.path.join(args.output_path, "MCMICRO.report")
    if not os.path.exists(output):
        os.makedirs(output)
    reference_folder = args.reference_folder
    markers_csv = args.markers_csv
    phenotypes_csv = args.phenotypes_csv
    gates = args.gates
    alternative_roi_path = args.alternative_roi_path
    area_csv = 'sample_area.csv'
    tumor_csv = 'tumor_area.csv'
    
    # Check files and directories
    print("Checking files and directories …")
    files_to_check = [h5ad_path, HnE_path, markers_csv, phenotypes_csv]
    directories_to_check = [output, reference_folder]
    for file in files_to_check:
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            sys.exit(1)
    for directory in directories_to_check:
        if not os.path.isdir(directory):
            print(f"Directory not found: {directory}")
            sys.exit(1)
    print("Files and directories checked.")
    
    workflow_types = {
        'Marker': 'Marker-based workflow',
        'HnE': 'H&E segmentation workflow',
        'Human': 'Human annotation workflow'
    }
    if workflow_type_key not in workflow_types:
        print("No valid workflow type detected")
        sys.exit(1)
    workflow_type = workflow_types[workflow_type_key]
    print(f"Selected workflow: {workflow_type}")
    if workflow_type_key == 'Marker':
        print("Tumor marker:", tumor_marker)
    
    # ---------------------------
    # Cell Phenotyping
    # ---------------------------
    print("\n...Running cell phenotyping…")
    markers = pd.read_csv(markers_csv)
    phenotypes = pd.read_csv(phenotypes_csv)
    missing_markers = set(markers['marker_name']) - set(phenotypes.columns)
    if missing_markers:
        print(f"Missing markers: {', '.join(missing_markers)}")
    else:
        print("All markers are present.")
    if gates:
        print("Reading gates file …")
        gates_df = pd.read_csv(gates)
        print("Reading single cell table …")
        adata = scimap.pp.mcmicro_to_scimap(h5ad_path, log=False, remove_dna=False)
        print("Running scimap rescaling …")
        adata = scimap.pp.rescale(adata, gate=gates_df, log=False)
        print("Running scimap phenotyping …")
        adata = scimap.tl.phenotype_cells(adata, phenotype=phenotypes, gate=0.5, label="phenotype")
    elif 'cspotPredict' in h5ad_path:
        print("File contains 'cspotPredict'; running phenotyping workflow …")
        adata = ad.read_h5ad(h5ad_path)
        adata = scimap.tl.phenotype_cells(adata, phenotype=phenotypes, gate=0.5, label="phenotype")
    else:
        print("No gates file provided and .h5ad file is not a cspotOutput file. Exiting.")
        sys.exit(1)
    
    # ---------------------------
    # Sample Detection & ROI Extraction
    # ---------------------------
    print("\n...Finding sample…")
    print(f"Loaded {adata.n_obs} cells from .h5ad file")
    idx = np.arange(0, adata.n_obs, 5)
    print(f"Using every {idx[1]}th cell for analysis.")
    adata_subsampled = adata[idx].copy()
    print("Extracting coordinates …")
    x_coords = adata_subsampled.obs['X_centroid'].values
    y_coords = adata_subsampled.obs['Y_centroid'].values
    coordinates = np.column_stack((x_coords, y_coords))
    print("Running DBSCAN to extract tissue areas …")
    eps = 2000
    min_samples = 100
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = db.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"DBSCAN found {n_clusters} cluster(s) plus noise.")
    valid_labels = [lbl for lbl in unique_labels if lbl != -1]
    if not valid_labels:
        print("No clusters found (all noise?)")
        sys.exit(1)
    else:
        print(f"Processing {len(valid_labels)} cluster(s).")
    alpha_value = 0.0005
    print(f"Using alpha = {alpha_value} for cluster outlines.")
    
    # Get pixel size from H&E image
    Image.MAX_IMAGE_PIXELS = None
    pixel_size_x = None
    with Image.open(HnE_path) as img:
        meta_dict = {TAGS.get(key, key): img.tag_v2[key] for key in img.tag_v2.keys()}
        description_str = meta_dict.get('ImageDescription', '')
        if isinstance(description_str, bytes):
            description_str = description_str.decode('utf-8', errors='ignore')
        match_x = re.search(r'PhysicalSizeX="([0-9.]+)"', description_str)
        if match_x:
            pixel_size_x = float(match_x.group(1))
    print(f"Pixel size X: {pixel_size_x} µm")
    
    all_polygons = []
    rows = []
    cluster_counter = 0
    for lbl in valid_labels:
        cluster_coords = coordinates[labels == lbl]
        shp = alphashape.alphashape(cluster_coords, alpha_value)
        if shp.is_empty:
            print(f"Cluster {lbl}: alpha shape is EMPTY, skipping.")
            continue
        if shp.geom_type == "Polygon":
            polygons = [shp]
        elif shp.geom_type == "MultiPolygon":
            polygons = list(shp.geoms)
        else:
            polygons = [g for g in shp.geoms if g.geom_type == 'Polygon']
        if not polygons:
            print(f"Cluster {lbl}: no valid polygons, skipping.")
            continue
        for poly in polygons:
            area_px, area_um2, area_mm2 = compute_areas(poly, pixel_size_x)
            if area_mm2 is not None and area_mm2 < 0.5:
                continue
            cluster_counter += 1
            coords_str = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in poly.exterior.coords)
            name_str = f"sample_area_{cluster_counter}"
            rows.append({
                "Id": cluster_counter,
                "Name": name_str,
                "Text": name_str,
                "type": "Polygon",
                "all_points": coords_str,
                "area_px": area_px,
                "area_um2": area_um2,
                "area_mm2": area_mm2
            })
    if not rows:
        print("No polygons ≥ 0.5 mm² found. Exiting.")
        sys.exit(1)
    df_sample = pd.DataFrame(rows, columns=["Id", "Name", "Text", "type", "all_points", "area_px", "area_um2", "area_mm2"])
    sample_area_csv_path = os.path.join(output, area_csv)
    df_sample.to_csv(sample_area_csv_path, index=False)
    print(f"Saved {len(rows)} sample polygon(s) to CSV: {sample_area_csv_path}")
    
    # Tumor detection based on workflow type
    if workflow_type_key == "Marker":
        print("\n...Finding tumor area…")
        adata_subsampled = scimap.hl.classify(adata_subsampled, pos=tumor_marker, classify_label=True, failed_label=False, label='Tumor_pos')
        pos_cells_tumor = adata_subsampled.obs[adata_subsampled.obs['Tumor_pos']]
        x_coords_tumor = pos_cells_tumor['X_centroid'].values
        y_coords_tumor = pos_cells_tumor['Y_centroid'].values
        coordinates_tumor = np.column_stack((x_coords_tumor, y_coords_tumor))
        print("Running DBSCAN for tumor area …")
        eps_tumor = 2000
        min_samples_tumor = 200
        db_tumor = DBSCAN(eps=eps_tumor, min_samples=min_samples_tumor).fit(coordinates_tumor)
        labels_tumor = db_tumor.labels_
        unique_labels_tumor = set(labels_tumor)
        n_clusters_tumor = len(unique_labels_tumor) - (1 if -1 in unique_labels_tumor else 0)
        print(f"DBSCAN found {n_clusters_tumor} tumor cluster(s) plus noise.")
        valid_labels_tumor = [lbl for lbl in unique_labels_tumor if lbl != -1]
        if not valid_labels_tumor:
            print("No tumor clusters found (all noise?). Exiting.")
            sys.exit(1)
        print(f"Using alpha = {alpha_value_tumor} for tumor clusters.")
        rows_tumor = []
        tumor_poly_counter = 0
        for lbl in valid_labels_tumor:
            print(f"Processing tumor cluster {lbl} …")
            cluster_coords_tumor = coordinates_tumor[labels_tumor == lbl]
            shp_tumor = alphashape.alphashape(cluster_coords_tumor, alpha_value_tumor)
            if shp_tumor.is_empty:
                print(f"Tumor cluster {lbl}: alpha shape is EMPTY, skipping.")
                continue
            if shp_tumor.geom_type == "Polygon":
                tumor_polygons = [shp_tumor]
            elif shp_tumor.geom_type == "MultiPolygon":
                tumor_polygons = list(shp_tumor.geoms)
            else:
                tumor_polygons = [g for g in shp_tumor.geoms if g.geom_type == 'Polygon']
            if not tumor_polygons:
                print(f"Tumor cluster {lbl}: no valid polygons, skipping.")
                continue
            for poly in tumor_polygons:
                tumor_poly_counter += 1
                area_px, area_um2, area_mm2 = compute_areas(poly, pixel_size_x)
                if area_mm2 is not None and area_mm2 < 0.5:
                    continue
                coords_str = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in poly.exterior.coords)
                name_str = f"tumor_area_{tumor_poly_counter}"
                rows_tumor.append({
                    "Id": tumor_poly_counter,
                    "Name": name_str,
                    "Text": name_str,
                    "type": "Polygon",
                    "all_points": coords_str,
                    "area_px": area_px,
                    "area_um2": area_um2,
                    "area_mm2": area_mm2
                })
        if not rows_tumor:
            print("No valid tumor polygons found. Exiting.")
            sys.exit(1)
        df_tumor = pd.DataFrame(rows_tumor, columns=["Id", "Name", "Text", "type", "all_points", "area_px", "area_um2", "area_mm2"])
        tumor_area_csv_path = os.path.join(output, tumor_csv)
        df_tumor.to_csv(tumor_area_csv_path, index=False)
        print(f"Saved {len(rows_tumor)} tumor polygon(s) to CSV: {tumor_area_csv_path}")
    elif workflow_type_key in ["HnE", "Human"]:
        print("Exporting tumor area from alternative ROI segmentation …")
        alternative_roi = pd.read_csv(alternative_roi_path)
        tumor_roi_raw = alternative_roi[alternative_roi['Text'].str.contains("tumor", regex=False, case=False, na=False)].copy()
        tumor_poly_counter = 0
        rows_tumor = []
        for idx, row in tumor_roi_raw.iterrows():
            coords = parse_polygon_coords(row['all_points'])
            poly = shapely.geometry.Polygon(coords)
            area_px, area_um2, area_mm2 = compute_areas(poly, pixel_size_x)
            if area_mm2 is not None and area_mm2 < 0.5:
                continue
            tumor_poly_counter += 1
            name_str = f"tumor_area_{tumor_poly_counter}"
            rows_tumor.append({
                "Id": tumor_poly_counter,
                "Name": name_str,
                "Text": row['Text'],
                "type": "Polygon",
                "all_points": row['all_points'],
                "area_px": area_px,
                "area_um2": area_um2,
                "area_mm2": area_mm2
            })
        if not rows_tumor:
            print("No valid tumor polygons found in alternative ROI. Exiting.")
            sys.exit(1)
        df_tumor = pd.DataFrame(rows_tumor, columns=["Id", "Name", "Text", "type", "all_points", "area_px", "area_um2", "area_mm2"])
        tumor_area_csv_path = os.path.join(output, tumor_csv)
        df_tumor.to_csv(tumor_area_csv_path, index=False)
        print(f"Saved {len(rows_tumor)} tumor polygon(s) to CSV: {tumor_area_csv_path}")
    
    # ---------------------------
    # Calculate Areas
    # ---------------------------
    print("\n...Calculating areas…")
    tumor_roi = pd.read_csv(os.path.join(output, tumor_csv))
    sample_roi = pd.read_csv(os.path.join(output, area_csv))
    tumor_roi["Name"] = "Tumor area"
    tumor_area_mm2 = round(tumor_roi["area_mm2"].sum(), 2)
    sample_area_mm2 = round(sample_roi["area_mm2"].sum(), 2)
    other_area_mm2 = round(sample_area_mm2 - tumor_area_mm2, 2)
    print(f"Total sample area: {sample_area_mm2} mm²")
    print(f"Total tumor area: {tumor_area_mm2} mm²")
    print(f"Other area: {other_area_mm2} mm²")
    
    # ---------------------------
    # Add ROI to AnnData
    # ---------------------------
    print("\n...Importing ROI…")
    adata = addROI_omero(adata, roi=tumor_roi, label='Tumor_ROI', overwrite=False)
    print("ROI added.")
    
    # ---------------------------
    # Build Cell Tree & Calculate Counts, Densities, Ratios
    # ---------------------------
    print("\n...Building cell tree and calculating counts…")
    phenotypes.rename(columns={phenotypes.columns[0]: "Parent", phenotypes.columns[1]: "Child"}, inplace=True)
    adjacency, all_nodes = build_adjacency(phenotypes)
    phenotype_col = "phenotype"
    observed_cell_types = set(adata.obs[phenotype_col].dropna().unique())
    missing_types = observed_cell_types - all_nodes
    if missing_types:
        print("The following cell types are missing from the tree:", missing_types)
        for cell_type in missing_types:
            adjacency["all"].append(cell_type)
            all_nodes.add(cell_type)
    else:
        print("No missing cell types found.")
    df_counts = compute_counts_node_and_roi(adata, adjacency, all_nodes,
                                              phenotype_col="phenotype",
                                              roi_col="Tumor_ROI")
    order = get_hierarchical_order(adjacency, "all")
    df_counts['Cell type'] = pd.Categorical(df_counts['Cell type'], categories=order, ordered=True)
    df_counts = df_counts.sort_values('Cell type')
    try:
        df_counts = df_counts[['Cell type', 'Total', 'Tumor area', 'Other']]
    except Exception as e:
        print("Warning: Expected ROI columns not found in cell counts.")
    df_counts.to_csv(os.path.join(output, "cell_counts.csv"), index=False)
    print("Cell counts saved.")
    print("Calculating cell densities per mm² …")
    df_cells_mm2 = df_counts.copy()
    df_cells_mm2['Total'] = round(df_cells_mm2['Total'].astype(float) / sample_area_mm2, 2)
    df_cells_mm2['Tumor area'] = round(df_cells_mm2['Tumor area'].astype(float) / tumor_area_mm2, 2)
    df_cells_mm2['Other'] = round(df_cells_mm2['Other'].astype(float) / other_area_mm2, 2)
    df_cells_mm2.to_csv(os.path.join(output, "cell_densities.csv"), index=False)
    print("Cell densities saved.")
    print("Calculating cell ratios …")
    df_cells_ratios = df_counts.copy()
    df_cells_ratios['Tumor area'] = round(df_cells_ratios['Tumor area'].astype(float) / df_cells_ratios['Total'].astype(float) * 100, 2)
    df_cells_ratios['Other'] = round(df_cells_ratios['Other'].astype(float) / df_cells_ratios['Total'].astype(float) * 100, 2)
    df_cells_ratios = df_cells_ratios[['Cell type', 'Tumor area', 'Other']]
    df_cells_ratios.to_csv(os.path.join(output, "cell_ratios.csv"), index=False)
    print("Cell ratios saved.")
    
    # ---------------------------
    # Generate Plots
    # ---------------------------
    print("\n...Generating plots…")
    df_counts['Tumor area'] = round(df_counts['Tumor area'].astype(float), 2)
    df_counts['Other'] = round(df_counts['Other'].astype(float), 2)
    sample_color = '#92c5de'
    tumor_color = '#d6604d'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    cell_types = df_cells_ratios['Cell type'].unique()
    axs[0].barh(cell_types, df_cells_ratios.groupby('Cell type')['Tumor area'].sum(),
                label='Tumor area', color=tumor_color)
    axs[0].barh(cell_types, df_cells_ratios.groupby('Cell type')['Other'].sum(),
                left=df_cells_ratios.groupby('Cell type')['Tumor area'].sum(),
                label='Other', color=sample_color)
    axs[0].set_title('Cell ratios (%)', size=12, loc='left')
    axs[0].legend()
    axs[0].invert_yaxis()
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[1].barh(cell_types, df_counts['Tumor area'], label='Tumor area', color=tumor_color)
    axs[1].barh(cell_types, df_counts['Other'], left=df_counts['Tumor area'], label='Other', color=sample_color)
    axs[1].set_title('Cell Counts (log10)', size=12, loc='left')
    axs[1].invert_yaxis()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].set_yticks([])
    axs[1].set_xscale('log')
    fig.tight_layout()
    plot_filename = "cell_ratios_and_counts_plot.png"
    plt.savefig(os.path.join(output, plot_filename), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined cell plot to {os.path.join(output, plot_filename)}")
    
    print("\n...Generating ROI plots…")
    h_and_e = tifffile.imread(HnE_path)
    plot_h_and_e_and_two_markers(h_and_e=h_and_e, image_path=ome_path, marker_csv=markers_csv,
                                 marker_1_name=sample_marker, marker_2_name=tumor_marker,
                                 sample_df=sample_roi, tumor_df=df_tumor, output_dir=output,
                                 pixel_size_x=pixel_size_x, workflow_type_key=workflow_type_key, sz=10)
    
    # ---------------------------
    # Immunoprofile (if enabled)
    # ---------------------------
    if immunoProfile:
        buffer_dist_microns = 40.0
        buffer_dist_pixels = buffer_dist_microns / pixel_size_x
        sample_polygons = []
        for _, row in sample_roi.iterrows():
            coords = parse_polygon_coords(row["all_points"])
            shp = Polygon(coords)
            if shp.is_valid:
                sample_polygons.append(shp)
            else:
                print(f"Warning: invalid polygon in sample row {row['Id']}.")
        sample_union = unary_union(sample_polygons)
        if sample_union.is_empty:
            print("No valid sample polygons found. Exiting.")
            sys.exit(1)
        tumor_polygons = []
        for _, row in df_tumor.iterrows():
            coords = parse_polygon_coords(row["all_points"])
            shp = Polygon(coords)
            if shp.is_valid:
                tumor_polygons.append(shp)
            else:
                print(f"Warning: invalid tumor polygon in row {row['Id']}.")
        if not tumor_polygons:
            print("No valid tumor polygons found. Exiting.")
            sys.exit(1)
        tumor_bands = []
        for shp in tumor_polygons:
            expanded_poly = shp.buffer(+buffer_dist_pixels)
            contracted_poly = shp.buffer(-buffer_dist_pixels)
            band_poly = expanded_poly.difference(contracted_poly)
            band_poly = band_poly.intersection(sample_union)
            if not band_poly.is_empty:
                tumor_bands.append(band_poly)
        if not tumor_bands:
            print("No valid tumor bands found. Exiting.")
            sys.exit(1)
        final_band = unary_union(tumor_bands)
        def total_area_mm2(poly, px_size):
            area_px = poly.area
            area_um2 = area_px * (px_size**2)
            return area_um2 / 1e6
        tumor_area_mm2 = sum(total_area_mm2(shp, pixel_size_x) for shp in tumor_polygons)
        expanded_polys = [shp.buffer(+buffer_dist_pixels) for shp in tumor_polygons]
        tumor_plus_tsi_poly = unary_union(expanded_polys)
        expanded_area_mm2 = total_area_mm2(tumor_plus_tsi_poly, pixel_size_x)
        band_area_mm2 = total_area_mm2(final_band, pixel_size_x)
        print(f"Tumor total area (mm²): {tumor_area_mm2:.5f}")
        print(f"Expanded tumor +{buffer_dist_microns}µm area (mm²): {expanded_area_mm2:.5f}")
        print(f"±{buffer_dist_microns}µm Tumor-Stroma interface (TSI) area (mm²): {band_area_mm2:.5f}")
        rows_band = []
        current_id = 100
        def polygon_to_coords_str(poly):
            return " ".join(f"{x:.2f},{y:.2f}" for (x, y) in poly.exterior.coords)
        if not final_band.is_empty:
            if final_band.geom_type == "Polygon":
                polygons_band = [final_band]
            elif final_band.geom_type == "MultiPolygon":
                polygons_band = list(final_band.geoms)
                print("Multiple sub-polygons in one band region.")
            else:
                polygons_band = [g for g in final_band.geoms if g.geom_type == "Polygon"]
            for poly in polygons_band:
                if poly.is_empty:
                    continue
                area_px = poly.area
                area_um2 = area_px * (pixel_size_x**2)
                area_mm2 = area_um2 / 1e6
                coords_str = polygon_to_coords_str(poly)
                rows_band.append({
                    "Id": current_id,
                    "Name": "band_tumor_roi",
                    "Text": "band_tumor_roi",
                    "type": "Polygon",
                    "all_points": coords_str,
                    "area_px": area_px,
                    "area_um2": area_um2,
                    "area_mm2": area_mm2
                })
                current_id += 1
        else:
            print("Warning: final_band is empty.")
        df_band_tumor = pd.DataFrame(rows_band, columns=["Id","Name","Text","type","all_points","area_px","area_um2","area_mm2"])
        output_band = os.path.join(output, "band_tumor_roi.csv")
        df_band_tumor.to_csv(output_band, index=False)
        print(f"Saved {len(df_band_tumor)} band tumor polygon(s) to CSV: {output_band}")
        for row in rows_band:
            print(f"Polygon ID={row['Id']} => area_px={row['area_px']}, area_um2={row['area_um2']}, area_mm2={row['area_mm2']}")
        adata = addROI_shapely(adata, band_poly=final_band, label='TSI', xcol='X_centroid', ycol='Y_centroid')
        print("Cells in Tumor-Stroma interphase (TSI):", adata.obs['TSI'].value_counts().get('TSI', 0))
        adata = addROI_shapely(adata, band_poly=tumor_plus_tsi_poly, label='Tumor+TSI', xcol='X_centroid', ycol='Y_centroid')
        print("Cells in Tumor+TSI:", adata.obs['Tumor+TSI'].value_counts().get('Tumor+TSI', 0))
        adata = addROI_shapely(adata, band_poly=tumor_polygons, label='Tumor', xcol='X_centroid', ycol='Y_centroid')
        print("Cells in Tumor:", adata.obs['Tumor'].value_counts().get('Tumor', 0))
        adata = scimap.hl.classify(adata, pos="CD8a", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='CD8a+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos="PD-1", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='PD-1+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos=["CD8a", "PD-1"], neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='CD8a+ PD1+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos="PD-1", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='PD-1+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos="FOXP3", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='FOXP3+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos="Pan-CK", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='Tumor Cells', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos="PD-L1", neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='PD-L1+', showPhenotypeLabel=False)
        adata = scimap.hl.classify(adata, pos=["PD-L1", "Pan-CK"], neg=None, classify_label='pos', failed_label='neg',
                                    threshold=0.5, collapse_failed=True, label='PD-L1+ Tumor Cells', showPhenotypeLabel=False)
        count_dict = {}
        density_dict = {}
        roi_names = ['TSI', 'Tumor+TSI', 'Tumor']
        roi_areas = {'TSI': band_area_mm2, 'Tumor+TSI': expanded_area_mm2, 'Tumor': tumor_area_mm2}
        phenotypes_list = ["Tumor Cells", "CD8a+", "PD-1+", "CD8a+ PD1+", "FOXP3+", "PD-L1+", "PD-L1+ Tumor Cells"]
        for pheno in phenotypes_list:
            counts = {}
            densities = {}
            for roi in roi_names:
                roi_cells = adata.obs[adata.obs[roi] == roi]
                cell_count = (roi_cells[pheno] == "pos").sum()
                counts[roi] = cell_count
                densities[roi] = cell_count / roi_areas[roi] if roi_areas[roi] else 0
            count_dict[pheno] = counts
            density_dict[pheno] = densities
        score_dict = {"TPS": {}, "CPS": {}}
        tumor_count = count_dict["Tumor Cells"]['Tumor']
        if tumor_count > 0:
            tps = (count_dict["PD-L1+ Tumor Cells"]['Tumor'] / tumor_count) * 100
            cps = (count_dict["PD-L1+"]['Tumor'] / tumor_count) * 100
        else:
            tps, cps = 0, 0
        score_dict["TPS"]['Tumor'] = tps
        score_dict["CPS"]['Tumor'] = cps
        immunoscore_df = pd.DataFrame(density_dict).T
        immunoscore_df.loc["Tumor Proportion Score (%)"] = pd.Series(score_dict["TPS"])
        immunoscore_df.loc["Combined Positive Score (%)"] = pd.Series(score_dict["CPS"])
        immunoscore_df.index.name = " "
        immunoscore_df = immunoscore_df.round(1)
        immunoscore_df = immunoscore_df.fillna('')
        immunoscore_df.to_csv(os.path.join(output, "cell_densities_and_scores.csv"))
    
    # ---------------------------
    # Create PDF Report
    # ---------------------------
    print("\n...Creating PDF report…")
    pdf = MyPDF('P', 'mm', 'Letter')
    pdf.reference_folder = reference_folder
    pdf.author = author
    # For this example, an empty DataFrame is passed for immunoscore_df; adjust as needed.
    create_pdf(pdf, output, sampleid, df_counts, workflow_type, tumor_marker, df_cells_mm2, pd.DataFrame(), author)
    
    end_time = time.time()
    print(f"Script running time: {round((end_time - start_time) / 60, 2)} minutes")
    print("Complete.")

if __name__ == '__main__':
    main()

