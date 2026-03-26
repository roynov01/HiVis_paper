# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 14:29:56 2026

@author: royno
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tifffile
from scipy import sparse
import anndata as ad
from HiVis import HiVis
import matplotlib as mpl

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


# Data was downloaded from: https://spateo-release.readthedocs.io/en/latest/tutorials/notebooks/1_cell_segmentation/stain_segmentation.html
#%% Import data
path_output = r"X:\roy\viziumHD\analysis\Python\version_11\stereoseq\spateo_tutorial\output"
image_file = r"X:\roy\viziumHD\analysis\Python\version_11\stereoseq\spateo_tutorial\data\SS200000135IL-D1.ssDNA.tif"
transcripts_file = r"X:\roy\viziumHD\analysis\Python\version_11\stereoseq\spateo_tutorial\data\SS200000135TL_D1_all_bin1.txt.gz"

fluorescence = {"ssDNA": "white"}
name = "stereoseq"
bin_size_um = 5
microns_per_pixel = 0.5 

brain = HiVis.new_stereoseq(
    path_transcripts=transcripts_file,
    path_image=image_file,
    bin_size_um=bin_size_um,  
    name=name,
    path_output=path_output,
    fluorescence=fluorescence,
    microns_per_pixel=microns_per_pixel,
    flip_img=True
)

ax = brain.plot.spatial("nUMI",exact=True,img_resolution="full",save=True,title="",show_zeros=1,scalebar={"text":False})
# brain.plot.spatial(exact=True,img_resolution="full",legend=False,save=True,scalebar={"text":False})

brain["nUMI_log10"] = np.log10(brain["nUMI"] +1)
ax = brain.plot.spatial("nUMI_log10",img_resolution="full",save=True,title="",
                        show_zeros=1,scalebar={"text":False},cmap="hot",exact=True,legend_title="log10(nUMI)")


brain.export_images()
#%% Add segmentation
brain.add_annotations(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\stereoseq_fullres_cells.geojson",name="SC",measurements=False)
brain.agg_from_annotations("SC_id", name="SC", obs2agg=["nUMI"])
brain.agg["SC"].import_geometry(r"X:\roy\viziumHD\analysis\Qupath\other_methods\results\stereoseq_fullres_cells.geojson",)
#%% Plot cells spatial
ax = brain.agg["SC"].plot.cells(line_color="lime",scalebar={"text":False},title="",linewidth=0.4)
brain.plot.save("cells",ax=ax)

# ax = brain.agg["SC"].plot.cells("Mbp",line_color="none",image=False,alpha=0.5,cmap="winter",xlim=[500,800],ylim=[500,800],scalebar={"text":False},title="")
# ax = brain.agg["SC"].plot.cells("Hpca",line_color="none",image=False,alpha=0.5,cmap="hot",xlim=[500,800],ylim=[500,800],scalebar={"text":False},title="",ax=ax)



ax = brain.agg["SC"].plot.cells(line_color="magenta",scalebar={"length":10,"text":False},
                                title="",xlim=[500,800],ylim=[500,800])
brain.plot.save("cells_blow_up",ax=ax)
#%% Celltype annotations
#%%% UMAP
import scanpy as sc

adata = brain.agg["SC"].adata.copy()

X = adata.X
    
cell_sums = np.asarray(X.sum(1)).ravel()
cell_sums[cell_sums==0] = 1
matnorm = X.multiply(1/cell_sums[:,None])
adata.layers["matnorm"] = matnorm

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

rbc_genes =  ['Hbb-bs', 'Hbb-bt', 'Hba-a1', 'Hba-a2']
sc.tl.score_genes(adata,gene_list=rbc_genes,score_name="rbc_score",use_raw=False)
adata.obs["rbc_score"].plot.hist(bins=50) 
if sparse.issparse(adata.X):
    adata.X = adata.X.tocsr()
for k in adata.layers.keys():
    if sparse.issparse(adata.layers[k]):
        adata.layers[k] = adata.layers[k].tocsr()
adata = adata[adata.obs["rbc_score"] < 0,:].copy()



sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=5)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8, key_added='leiden')
sc.pl.umap(adata, color=['leiden'])
#%%% Assign clusters
import math
import scipy.sparse as sp

def assign_clusters(adata, genes_map, new_col_name="celltype_lvl1", cluster_col="leiden", 
                    ensure_coverage=True, plot=True, layer=None):
    if cluster_col not in adata.obs:
        raise KeyError(f"'{cluster_col}' not found in adata.obs.")

    # If no mapping provided: just copy cluster ids into new_col_name (no plotting here)
    if not genes_map:
        adata.obs[new_col_name] = adata.obs[cluster_col].astype(str).astype("category")
        return

    X = adata.X if layer is None else adata.layers[layer]
    x_mean = float(X.mean()) if sp.issparse(X) else float(np.mean(X))
    if not np.isclose(x_mean, 0.0, atol=0.1):
        raise ValueError("adata.X (or selected layer) isn't scaled (mean not ~ 0).")

    # normalize mapping to lists and keep only genes present
    norm_map = {}
    for ct, genes in genes_map.items():
        genes = [genes] if isinstance(genes, str) else list(genes)
        present = [g for g in genes if g in adata.var_names]
        if present:
            norm_map[ct] = present
    if not norm_map:
        raise ValueError("None of the provided marker genes are in adata.var_names.")

    all_markers = sorted({g for glist in norm_map.values() for g in glist})

    expr_df = sc.get.obs_df(adata, keys=all_markers, layer=layer, use_raw=False).copy()
    expr_df[cluster_col] = adata.obs[cluster_col].astype(str).to_numpy()

    gene_means = expr_df.groupby(cluster_col, observed=True).mean(numeric_only=True)
    gene_means = gene_means[all_markers]

    ct_scores = pd.DataFrame(index=gene_means.index)
    for ct, genes in norm_map.items():
        ct_scores[ct] = gene_means[genes].max(axis=1)

    best_ct_per_cluster = ct_scores.idxmax(axis=1)

    if ensure_coverage:
        all_cts = set(norm_map.keys())
        present_cts = set(best_ct_per_cluster.dropna().unique())
        n_clusters = len(best_ct_per_cluster)
        if len(all_cts) > n_clusters:
            print(f"Requested {len(all_cts)} cell types but only {n_clusters} clusters; full coverage is impossible.")

        missing_cts = sorted(all_cts - present_cts)
        counts = best_ct_per_cluster.value_counts().to_dict()
        locked_clusters = set()

        for ct in missing_cts:
            series = ct_scores[ct].sort_values(ascending=False, kind="mergesort")
            placed = False
            for c, _val in series.items():
                if c in locked_clusters:
                    continue
                donor = best_ct_per_cluster.loc[c]
                if isinstance(donor, float) and math.isnan(donor):
                    donor = None
                if donor is None or counts.get(donor, 0) > 1:
                    if donor is not None:
                        counts[donor] = counts.get(donor, 0) - 1
                    best_ct_per_cluster.loc[c] = ct
                    counts[ct] = counts.get(ct, 0) + 1
                    locked_clusters.add(c)
                    placed = True
                    break
            if not placed:
                print(f"Could not assign any cluster to missing cell type '{ct}' without stealing the last cluster of another type.")

    adata.obs[new_col_name] = adata.obs[cluster_col].astype(str).map(best_ct_per_cluster).astype("category")

    # Plot only once, at the end, showing both cluster ids and new identities
    if plot:
        def _pick_one_marker_per_celltype(adata, genes_map):
            picked = {}
            for ct, genes in genes_map.items():
                if genes is None:
                    continue
                if isinstance(genes, str):
                    genes = [genes]
                elif isinstance(genes, dict):
                    genes = list(genes.keys())
                else:
                    genes = list(genes)
                genes = [g for g in genes if isinstance(g, str) and len(g) > 0]
                present = [g for g in genes if g in adata.var_names]
                if present:
                    picked[ct] = present[0]
            return picked
        picked = _pick_one_marker_per_celltype(adata, genes_map)
        gene_list = list(dict.fromkeys(picked.values()))  # preserve order, unique
        sc.pl.umap(adata, show=False, color=[cluster_col, new_col_name] + gene_list, legend_loc="on data", palette="tab20")


genes_map = {"Astrocyte":["Slc1a2","Slc1a3","Aldh1l1","Aqp4","Sox9","Apoe"],
  "Oligodendrocyte":["Plp1","Mbp","Mobp","Mag","Mog","Cnp","Sox10"],
  "Neuron":["Rbfox3","Snap25","Syt1","Tubb3","Map2","Slc17a7","Gad1","Hpca"],
  "RBC": ['Hbb-bs', 'Hbb-bt', 'Hbb-y']
}


assign_clusters(adata, genes_map, new_col_name="celltype", cluster_col="leiden")

sc.pl.umap(adata, show=False, color=["celltype"], legend_loc="on data",legend_fontsize=10,palette="tab20")


#%%% Plot spatial celltypes
brain.agg["SC"].merge(adata,obs=["celltype"])

ax = brain.agg["SC"].plot.cells("temp",line_color="none",scalebar={"text":False},title="",
                                save=True,legend_title=False,legend="upper left",
                                cmap={'RBC':"maroon","Neuron":"darkblue","Astrocyte":"darkgreen",
                                      "Oligodendrocyte":"darkgoldenrod","Microglia":"gray"},
                                image=False)

brain.plot.save("celltypes",ax=ax)

