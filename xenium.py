# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:56:54 2026

@author: royno
"""
import numpy as np
import pandas as pd
from scipy import sparse
from HiVis import HiVis
from scipy.stats import spearmanr, wilcoxon
from tqdm import tqdm
import scanpy as sc
import  matplotlib.pyplot as plt
import matplotlib as mpl
import anndata as ad
from tifffile import tifffile
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    "font.size": 14,     
    "axes.titlesize": 16,    
    "axes.labelsize": 14,  
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,  
    "legend.fontsize": 14 
})

def load_xenium_image(xenium_outs_path):
    from pathlib import Path

    xenium_outs_path = Path(xenium_outs_path)
    morph_path = xenium_outs_path / "morphology_focus"
    if not morph_path.exists():
        raise FileNotFoundError(f"Cannot find morphology_focus in {xenium_outs_path}")

    # 2. list morphology OME-TIFFs
    tiff_paths = sorted(morph_path.glob("*.ome.tif*"))
    if not tiff_paths:
        raise FileNotFoundError(f"No OME-TIFFs found in {morph_path}")

    selected_imgs = []
    selected_channel_names = []

    # 3. iterate over files, parse channel name, and load
    for path in tiff_paths:
        name = path.name

        # Strip ".ome.tif" or ".ome.tiff"
        base = name.split(".ome")[0]  # e.g. "ch0000_dapi" or "ch0001_atp1a1_cd45_e-cadherin"
        parts = base.split("_", 1)
        if len(parts) < 2:
            print(f"Warning: could not parse channel from filename '{name}', skipping.")
            continue

        channel_name = parts[1]  # everything after first "_"

        # Load the image
        arr = tifffile.imread(path)

        # Convert to 2D Y x X
        if arr.ndim == 2:
            img_yx = arr
        elif arr.ndim == 3:
            # Assumption: first axis is an extra dimension (e.g. scale); use first plane
            img_yx = arr[0]
        else:
            raise ValueError(
                f"Unexpected number of dimensions ({arr.ndim}) for file {name}; "
                "expected 2D or 3D."
            )

        # Check shape consistency across channels
        if selected_imgs:
            if img_yx.shape != selected_imgs[0].shape:
                raise ValueError(
                    f"Image shape mismatch for file {name}: {img_yx.shape} "
                    f"vs {selected_imgs[0].shape} from previous file(s)."
                )

        selected_imgs.append(img_yx)
        selected_channel_names.append(channel_name)

    if not selected_imgs:
        raise ValueError(
            "No images were loaded. All files failed channel parsing or there were none."
        )

    # 4. stack into Y x X x C
    img_yxc = np.stack(selected_imgs, axis=-1)

    print(f"\nFinal image shape (Y, X, C): {img_yxc.shape}")
    print("Channels included (in C order):")
    for i, ch in enumerate(selected_channel_names):
        print(f"  C[{i}] -> {ch}")
    return img_yxc




#%% Import data
bin_size_um = 3
images_path = r"X:\roy\viziumHD\analysis\Python\version_11\xenium_apicome\data\Virgin_ileum_rep1"
path_output = r"X:\roy\viziumHD\analysis\Python\version_11\xenium_apicome\output"
path = r"X:\roy\viziumHD\analysis\Python\version_11\xenium_apicome\data\Virgin_ileum_rep1\GSM8695241_TA1_transcripts.parquet"
microns_per_pixel = 0.2125
df = pd.read_parquet(path)

# 2. Rename required columns directly
df = df.rename(columns={"x_location": "x_um", "y_location": "y_um", "z": "z_um", "feature_name": "gene"})

# 3. Filters
df = df[df["qv"] >= 20]
ignore_rows = df["gene"].str.startswith("NegControl") | df["gene"].str.startswith("Unassigned")
df = df.loc[~ignore_rows]

# 4. Bounding box only for relative coordinates

x_min = df["x_um"].min()
y_min = df["y_um"].min()

df["x_rel_um"] = df["x_um"] - x_min
df["y_rel_um"] = df["y_um"] - y_min

# 5. Assign bins
df["bin_x"] = (df["x_rel_um"] / bin_size_um).astype(int)
df["bin_y"] = (df["y_rel_um"] / bin_size_um).astype(int)


# 6. Aggregate counts
grouped = df.groupby(["bin_x", "bin_y", "gene"]).size().reset_index(name="count")
matrix = grouped.pivot_table(index=["bin_x", "bin_y"], columns="gene", values="count", fill_value=0)
matrix = matrix.sort_index(axis=0).sort_index(axis=1)

# 7. Compute bin centers
bin_indices = np.array(matrix.index.tolist())
bin_x = bin_indices[:, 0]
bin_y = bin_indices[:, 1]

x_center_rel_um = (bin_x + 0.5) * bin_size_um
y_center_rel_um = (bin_y + 0.5) * bin_size_um

um_x = x_center_rel_um + x_min
um_y = y_center_rel_um + y_min


pxl_x = x_center_rel_um / microns_per_pixel
pxl_y = y_center_rel_um / microns_per_pixel

# 9. Build AnnData
obs_index = pd.Index([f"bin_x{i}_y{j}" for i, j in zip(bin_x, bin_y)], name="bin_id")
obs = pd.DataFrame({"um_x": um_x, "um_y": um_y, "pxl_row_in_fullres": pxl_y, "pxl_col_in_fullres": pxl_x}, index=obs_index)
var = pd.DataFrame(index=matrix.columns)
X = sparse.csr_matrix(matrix.values)

adata = ad.AnnData(X=X, obs=obs, var=var)
adata.uns["bin_size_um"] = bin_size_um
adata.uns["microns_per_pixel"] = microns_per_pixel



img = load_xenium_image(images_path)



high_res_scale = 0.25
low_res_scale = 0.01
fluorescence = {"DAPI":"white"}
downscaled_img, high_res_image, low_res_image, microns_per_pixel = HiVis.HiVis_utils.rescale_img_and_adata(adata,
                                                                microns_per_pixel, img,down_factor=1,
                                                                fluorescence=fluorescence,
                                                                high_res_scale=high_res_scale, low_res_scale=low_res_scale)

scalefactor_json = {"microns_per_pixel":microns_per_pixel,
                    "bin_size_um":bin_size_um,
                    "spot_diameter_fullres": bin_size_um/microns_per_pixel,
                    "tissue_hires_scalef": high_res_scale,
                    "tissue_lowres_scalef": low_res_scale}

properties = {}
HiVis.HiVis_utils._edit_adata(adata, scalefactor_json, "MT-")


rep1_full = HiVis.HiVis(adata, downscaled_img, high_res_image, low_res_image, scalefactor_json, 
         name="rep1", path_output=path_output,properties=properties, agg=None, fluorescence=fluorescence)


rep1 = rep1_full.crop(xlim=[300,2300],ylim=[2200,4200])

rep1.export_images(force=False)

#%%Import Lumen annotation
rep1.add_annotations(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_lumen.geojson",
                  name="lumen",measurements=False)
rep1.agg_from_annotations("lumen_id", name="lumen")
rep1.agg["lumen"].import_geometry(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_lumen.geojson",
                            object_type="annotation")

rep1.analysis.compute_distances("lumen")

rep1.add_annotations(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_muscle.geojson",
                  name="muscle",measurements=False)
rep1.agg_from_annotations("muscle_id", name="muscle")
rep1.agg["muscle"].import_geometry(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_muscle.geojson",
                            object_type="annotation")


xlim, ylim = [160,460], [925,1225]

rep1["temp"] = rep1["dist_to_lumen"]
rep1.adata.obs.loc[(rep1["lumen"]=="lumen"),"temp"] = np.nan

ax = rep1.plot.spatial("temp",legend=True,cmap=["blue","orange"], ylim=ylim,
                       xlim=xlim,exact=1,show_zeros=1,alpha=0.7,title="Distance from lumen")

raise StopAsyncIteration
#%% Single Cell
path = r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_cells_detections.csv"
input_df = pd.read_csv(path,sep="\t")
rep1.agg_cells(input_df,name="SC",obs2agg=["nUMI","dist_to_lumen"],geojson_path=path.replace("detections.csv","cells.geojson"))


rep1.add_annotations(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_nucs_cells.geojson",
                  name="nuc",measurements=False)
rep1.agg_from_annotations("nuc_id", name="nuc",obs2agg=["nUMI","dist_to_lumen"])
rep1.agg["nuc"].import_geometry(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\rep1_subset_nucs_cells.geojson",
                            object_type="detection")

rep1.analysis.compute_distances("nuc")

rep1.agg["nuc"].analysis.compute_distances("lumen")
rep1.agg["SC"].analysis.compute_distances("lumen")
rep1.agg["SC"].analysis.compute_distances("muscle")


#%% Assign epithelial cells
rep1.agg["SC"]["epi"] = (rep1.agg["SC"]["dist_to_rep1_subset_lumen"]<10) & (rep1.agg["SC"]["dist_to_rep1_subset_muscle"]>30)
xlim, ylim = [160,490], [930,1260]
ax = rep1.agg["SC"].plot.cells("epi",xlim=xlim,ylim=ylim,line_color="red",cmap=["blue","blue"],
                               title="Epithelial cells",legend=False,alpha=0.4)
# rep1.agg["nuc"].plot.cells(xlim=xlim,ylim=ylim,line_color="orange",legend=False,ax=ax,linewidth=1)
# HiVis.HiVis_plot.add_legend({"red":"Epithelial cells"}, ax)
rep1.plot.save("Epithelial_cells_spatial",ax=ax)


rep1.agg["SC"].sync("epi")
rep1.adata.obs.loc[rep1.adata.obs["epi"].isna(),"epi"] = False

# xlim, ylim = [150,450], [930,1260]
xlim, ylim = None, None

rep1.agg["SC"].plot.cells("epi",xlim=xlim,ylim=ylim,line_color="none",image=False,cmap=["red","red"])


#%% Apicome
#%%% Assign apicome
epi = rep1[rep1["epi"] & (~rep1.adata.obs["InNuc"].astype(bool)),:]

xlim, ylim = [0,250], [930,1260]
epi.plot.spatial("dist_to_lumen",xlim=xlim,ylim=ylim,exact=True)
epi.plot.hist("dist_to_lumen",bins=50)

epi.adata.obs["apicome"] = np.nan
epi.adata.obs.loc[epi["dist_to_lumen"]>15,"apicome"] = "basal"
epi.adata.obs.loc[epi["dist_to_lumen"]<10,"apicome"] = "apical"

# counts = epi.adata.obs.groupby("Cell_ID_SC")["apicome"].value_counts(dropna=True).unstack(fill_value=0)
# counts["has_apicome"] = (counts["apical"] > 0) 
# mask = epi.adata.obs["Cell_ID_SC"].map(counts["has_apicome"]).fillna(False)
# epi.adata.obs.loc[~mask, "apicome"] = np.nan



#%%% DGE
df = epi.analysis.dge("apicome", group1="apical", group2="basal",method="fisher_exact",two_sided=False) 
df["gene"] = df.index

apical_examples = ["Apoa1","Slc39a4"]
basal_examples = ["Igha","Mgam"]

qval_thresh = 0.1

plot = df.copy()
# plot = df.loc[df["gene"].isin(epi_genes)]

plot["expression_min"] = np.log10(plot["expression_min"])

apical_plot = plot.loc[plot["log2fc"] >= 0]
basal_plot = plot.loc[plot["log2fc"] < 0]

apical_genes = apical_plot.index[(apical_plot["qval"] < qval_thresh)]
apical_genes_text = apical_plot.index[(apical_plot["qval"] < qval_thresh) & apical_plot.index.isin(apical_examples) ]
apical_genes_text = apical_plot.index[(apical_plot["qval"] < qval_thresh)  ]
apical_genes_text = (apical_plot.loc[apical_plot["qval"] < qval_thresh]
                     .sort_values("qval")
                     # .head(30)
                     .index)

basal_genes = basal_plot.index[(basal_plot["qval"] < qval_thresh)]
basal_genes_text = basal_plot.index[(basal_plot["qval"] < qval_thresh) & basal_plot.index.isin(basal_examples) ]
basal_genes_text = basal_plot.index[(basal_plot["qval"] < qval_thresh) ]
basal_genes_text = (basal_plot.loc[basal_plot["qval"] < qval_thresh]
                     .sort_values("qval")
                     # .head(70)
                     .index)
# apical_genes_text = plot["gene"].tolist()

plt.rcParams.update({"font.size": 10})
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=apical_genes_text,color_genes="red",
                                          genes2=basal_genes_text,color_genes2="blue",repel=False,
                                          text=True, color="gray",y_line=0,figsize=(7,7))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", "log2fc",
                                          genes=apical_genes,color_genes="red",
                                          genes2=basal_genes,color_genes2="blue",
                                          text=False, color="gray",ax=ax,
                                          xlab="log10(expression)",ylab="log2(apical/basal)",
                                          title="Polarized genes")
plt.rcParams.update({"font.size": 14})

ax.set_ylim([-6,2])
ax.text(0.98, 0.95,"Apical mRNAs",transform=ax.transAxes,ha='right',color="red",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Basal mRNAs",transform=ax.transAxes,ha='right',color="blue",
        bbox=dict(facecolor='white', edgecolor='black'))

epi.plot.save("apicome_DGE",ax=ax)

#%% Nuc/cyto
#%%% Assign identities
epi2 = rep1[rep1.adata.obs["epi"].tolist(),:]


#%%% DGE
df = epi2.analysis.dge("InNuc", group1=1, group2=0,method="fisher_exact",two_sided=False) 
df["gene"] = df.index

#%%% MA
apical_examples = ["Neat1", "Vegfa"]
basal_examples = ["Fabp2","Apoa1"]

qval_thresh = 0.1

plot = df.copy()
# plot = df.loc[df["gene"].isin(epi_genes)]

plot["expression_min"] = np.log10(plot["expression_min"])

apical_plot = plot.loc[plot["log2fc"] >= 0]
basal_plot = plot.loc[plot["log2fc"] < 0]

nuc_genes = apical_plot.index[(apical_plot["qval"] < qval_thresh)]
nuc_genes_text = apical_plot.index[(apical_plot["qval"] < qval_thresh) & apical_plot.index.isin(apical_examples) ]
# nuc_genes_text = apical_plot.index[(apical_plot["qval"] < qval_thresh)]


cyto_genes = basal_plot.index[(basal_plot["qval"] < qval_thresh)]
cyto_genes_text = basal_plot.index[(basal_plot["qval"] < qval_thresh) & basal_plot.index.isin(basal_examples) ]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=nuc_genes_text,color_genes="darkcyan",
                                          genes2=cyto_genes_text,color_genes2="darkorange",repel=False,
                                          text=True, color="gray",y_line=0,figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", "log2fc",
                                          genes=nuc_genes,color_genes="darkcyan",
                                          genes2=cyto_genes,color_genes2="darkorange",
                                          text=False, color="gray",ax=ax,
                                          xlab="log10(expression)",ylab="log2(nucleus/cytoplasm)",
                                          title="Cytoplasmic bias - mouse intestine (Xenium)")

ax.text(0.98, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="darkcyan",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="darkorange",
        bbox=dict(facecolor='white', edgecolor='black'))

epi.plot.save("nuc_cyto_DGE",ax=ax)


#%% Plot spatial (apicome, image, cells)
#%%% Image
xlim, ylim = [0,330], [930,1260]

ax = rep1.plot.spatial(img_resolution="full",legend=False)
ax.set_title(None)

# add recktangle "zoom in"
from matplotlib.patches import Rectangle

mpp = rep1.json["microns_per_pixel"] 
xmin_um, xmax_um = xlim
ymin_um, ymax_um = ylim

xmin_px = xmin_um / mpp
xmax_px = xmax_um / mpp
ymin_px = ymin_um / mpp
ymax_px = ymax_um / mpp

rect = Rectangle((700, ymin_px), 1700, ymax_px - ymin_px, 
                 fill=False, edgecolor='white', linewidth=1)
ax.add_patch(rect)

rep1.plot.save("full_img_spatial",ax=ax)

xlim, ylim = [0,330], [930,1260]
ax = epi.plot.spatial(xlim=xlim,ylim=ylim,legend=False,img_resolution="full")
ax.set_title(None)


rep1.plot.save("ROI_spatial",ax=ax)

#%%% Apicome
epi["temp"] = epi.adata.obs["apicome"].str.capitalize()
xlim, ylim = [0,330], [930,1260]
ax = epi.plot.spatial("temp",xlim=xlim,ylim=ylim,exact=True,legend_title=False, legend="upper right", 
                 title="Bins assignment",cmap={"Apical":"red","Basal":"blue"})

rep1.plot.save("apicome_spatial",ax=ax)

#%%% Cells
xlim, ylim = [160,490], [930,1260]

ax = rep1.agg["nuc"].plot.cells(xlim=xlim,ylim=ylim,legend=False, line_color="lightgreen",
                 title="Mouse intestine (Xenium)",linewidth=0.5)
ax = rep1.agg["SC"].plot.cells(xlim=xlim,ylim=ylim,legend=False, line_color="red",
                 img_resolution="full",ax=ax,linewidth=0.5)
ax.set_title(None)

rep1.plot.save("cells",ax=ax)

#%%% Nuc/cyto
epi2["temp"] = epi2["InNuc"]
epi2.update_meta("temp",{1:"Nucleus",0:"Cytoplasm"})

xlim, ylim = [0,330], [930,1260]
ax = epi2.plot.spatial("temp",xlim=xlim,ylim=ylim,exact=True,legend_title=False, legend="upper right", 
                 title="Mouse intestine (Xenium)",cmap={"Nucleus":"cyan","Cytoplasm":"orange"},alpha=0.7)
ax = epi2.agg["nuc"].plot.cells(ax=ax,xlim=xlim,ylim=ylim,line_color="blue",linewidth=0.5)
rep1.plot.save("nuc_cyto_spatial",ax=ax)

#%%% Distance

xlim, ylim = [0,330], [930,1260]
ax = epi2.plot.spatial("dist_to_lumen",xlim=xlim,ylim=ylim,exact=True,legend_title="Distance (µm)",
                 title="Distance to lumen",cmap=["red","blue"],alpha=0.7)
rep1.plot.save("distance_to_lumen_spatial",ax=ax)

#%%% Epithelial cells

xlim, ylim = [160,490], [930,1260]

ax = rep1.agg["SC"].plot.cells("epi",xlim=xlim,ylim=ylim,legend=False, line_color="lime",
                 img_resolution="full",linewidth=0.4,cmap=["violet","violet"],title="Epithelial cells")
# ax = rep1.agg["nuc"].plot.cells(xlim=xlim,ylim=ylim,legend=False, line_color="black",ax=ax)
rep1.plot.save("epi_cells_spatial",ax=ax)

#%% UMAP
umi_thresh = 0
rep1.agg["SC"] = rep1.agg["SC"][rep1.agg["SC"]["nUMI"] >= umi_thresh,:]

adata = rep1.agg["SC"].adata.copy()


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["log_norm"] = adata.X.copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(adata,20)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)

sc.tl.leiden(adata, resolution=0.5, key_added='leiden')
sc.pl.umap(adata, color=['leiden'],legend_loc="on data")

sc.pl.umap(adata, color=['Lgr5','Epcam','Igha',"Top2a","Ada","Muc2","Defa17","Ptprc","Pecam1","Pdgfra","Acta2","Prph"])
['Lgr5','Muc2','Ptprc',"Top2a","Ada","Muc2","Defa17","Ptprc","Pecam1","Pdgfra","Acta2","Prph"]


sc.pl.umap(adata, color=['Aldob','leiden'],layer="log_norm")


#%%% Find markers

sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon')

sc.pl.rank_genes_groups(adata, n_genes=3, sharey=False)


markers = []
for cl in adata.obs['leiden'].cat.categories:
    genes = adata.uns['rank_genes_groups']['names'][cl][:4]
    markers.extend(genes)

markers = list(dict.fromkeys(markers))
markers = ['Lgr5','Muc2','Ptprc',"Top2a","Ada","Muc2","Defa22","Ptprc","Pecam1","Pdgfra","Acta2","Prph"]

# dot plot of markers per Leiden
sc.pl.dotplot(adata, markers, groupby='leiden', standard_scale='var')



markers = ["Fibroblast","1","2","3","4","5","6","7","8","9","10","11"]

sc.pl.umap(adata, color=['Col18a1','Igha','leiden'],layer="log_norm")



#%%% Investigate cluster identities

# ---- user inputs ----
groupby = "leiden"  # change to the column in adata.obs that holds these labels

# ---- canonical marker templates (will be intersected with your limited adata.var_names) ----
marker_sets_raw = {
    "paneth": ["LYZ1","DEFA17","DEFA22","DEFA5","DEFA6"],
    "stem": ["LGR5","OLFM4","CD24A"],
    "TA": ["MKI67","TOP2A"],
    "T": ["CD3E","GZMA"],
    "B": ["MS4A1","IGHA","CD19"],
    "fibroblast": ["COL1A1","COL1A2","DCN","PDGFRA"],
    "endothel": ["PECAM1","VWF","KDR","CDH5"],
    "muscle": ["ACTA2","MYH11","TAGLN","DES"],
    "neuron": ["ELAVL4","TUBB3","RBFOX3","RET","TAC1"],
    "ent_early": ["SIS",],
    "ent_mid": ["SLC5A1","SLC2A2","SLC2A5","LCT","PRAP1","RBP2","FABP1"],
    "end_mature": ["ALPI","APOA1","APOA4","PRAP1","ADA"],
    "goblet": ["MUC2","TFF3","CLCA1","KLF4"],
    "myeloid": ["PTPRC","ADGRE1","ITGAX","FCGR3A"],
}

# ---- make a copy; do NOT modify original adata ----
adata_sub = adata.copy()

# keep the plotting order stable

# ---- match markers to your (limited) var_names robustly (case-insensitive) ----
var_lower_to_var = {g.lower(): g for g in adata_sub.var_names}

marker_sets = {}
for ct, glist in marker_sets_raw.items():
    mapped = [var_lower_to_var[g.lower()] for g in glist if g.lower() in var_lower_to_var]
    marker_sets[ct] = sorted(set(mapped), key=mapped.index)

# optional sanity printout: how many markers survived per module
print({ct: len(glist) for ct, glist in marker_sets.items()})

# ---- score modules (writes into adata_sub.obs only) ----
score_cols = []
for ct, glist in marker_sets.items():
    score_name = f"score_{ct}"
    if len(glist) >= 1:
        sc.tl.score_genes(adata_sub, gene_list=glist, score_name=score_name, use_raw=adata_sub.raw is not None)
    else:
        adata_sub.obs[score_name] = np.nan
    score_cols.append(score_name)

# keep whatever module columns you already computed
module_cols = [c for c in adata_sub.obs.columns if c.startswith("score_")]

# mean score matrix: rows=leiden, cols=modules
mean_mat = adata_sub.obs.groupby(groupby)[module_cols].mean()

# define leiden order as it currently appears in the plot
# (categorical order if set; otherwise sorted unique values as strings)
if pd.api.types.is_categorical_dtype(adata_sub.obs[groupby]):
    leiden_order = list(adata_sub.obs[groupby].cat.categories.astype(str))
else:
    leiden_order = sorted(adata_sub.obs[groupby].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)

# map leiden -> position along x-axis
leiden_pos = {k: i for i, k in enumerate(leiden_order)}
mean_mat = mean_mat.reindex(index=leiden_order)

# for each module, where does it peak (which leiden)?
peak_leiden = mean_mat.idxmax(axis=0)
peak_pos = peak_leiden.astype(str).map(leiden_pos)

# sort modules by peak position (left -> right), then by peak height (stronger first)
peak_height = mean_mat.max(axis=0)
module_order_diag = (pd.DataFrame({"peak_pos": peak_pos, "peak_height": peak_height})
                     .sort_values(["peak_pos", "peak_height"], ascending=[True, False])
                     .index.tolist())

sc.pl.dotplot(adata_sub, var_names=module_order_diag, groupby=groupby, swap_axes=True, standard_scale="var")


#%%% Assign celltypes
leiden_to_celltype = {
    "0": "TA",
    "1": "Immune",
    "2": "Enterocyte early",
    "3": "Paneth",
    "4": "Goblet",
    "5": "Enterocyte late",
    "6": "Stem",
    "7": "Immune",
    "8": "Fibroblast",
    "9": "Endothel",
    "10": "Muscle",
    "11": "Neuron",
}


adata.obs["celltype"] = adata.obs["leiden"].astype(str).map(leiden_to_celltype)
# celltype_order = ["TA","B/Myeloid","Enterocyte early","Paneth","Goblet","Enterocyte late","Stem","T","Fibroblast","Endothel","Muscle","Neuron"]
celltype_order = ["Stem","Paneth","TA","Enterocyte early","Enterocyte late","Goblet",
                  "Immune","Fibroblast","Endothel","Muscle","Neuron"]

adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"], categories=celltype_order,  ordered=True)


#%%% Plot dotplot
markers = ["Olfm4","Defa17","Top2a","Sis","Apoa1","Tff3",
                 "Ptprc","Pdgfra","Pecam1","Acta2","Prph"]
# markers = ['Top2a','Itgax','Sis',"Defa17","Tff3","Apoa1","Olfm4","Cd3e","Pdgfra","Pecam1","Acta2","Prph"]

dp = sc.pl.dotplot(adata, var_names=markers, groupby="celltype",  standard_scale="var",
                   swap_axes=True,show=False,figsize=(6,6))

ax = dp["mainplot_ax"]
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
rep1.plot.save("dotplot",ax=ax)

#%%% Plot UMAP
celltype_cmap = {
    "Stem": "#1f77b4",               # blue
    "Paneth": "#ff7f0e",             # orange
    "TA": "#9467bd",                 # green
    "Enterocyte early": "lightgreen",   
    "Enterocyte late": "green",   
    "Goblet": "#8c564b",             # brown
    "Immune": "#e377c2",            # pink
    "Fibroblast": "#bcbd22",         # olive
    "Endothel": "#17becf",           # cyan
    "Muscle": "#aec7e8",             # light blue
    "Neuron": "#ffbb78",             # light orange
}

fig, ax = plt.subplots(figsize=(6,6))

sc.pl.umap(adata, color=['celltype'],legend_loc="on data", title=None,show=False,ax=ax,size=15,palette=celltype_cmap,)

for txt in ax.texts:
    txt.set_fontweight("normal")
    txt.set_fontsize(12)
ax.set_title(None)
rep1.plot.save("UMAP",ax=ax)

#%%% Plot spatial

rep1.agg["SC"].merge(adata,["celltype"])
xlim, ylim = [150,480], [930,1260]
ax = rep1.agg["SC"].plot.cells("celltype", xlim=xlim, ylim=ylim, cmap=celltype_cmap,
                               image=False, alpha=1,scalebar=False,legend=False)
ax.set_title(None)

rep1.plot.save("celltypes_SPATIAL",ax=ax)
