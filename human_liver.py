# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:18:33 2025

@author: royno
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from jenkspy import jenks_breaks

from HiVis import HiVis

#%% Import data
properties = {"organism":"human",
              "organ":"liver",
              "cancer": False,
              "sample_id":"human_liver_M2M6",
              "source":"roy"}
path_image_fullres = r"X:\roy\other_projects\Orans\data\M2M6_orig.tif"
path_input_data = r"X:\oran\Data\Human_Liver_Project\Human_Liver_Visium\Visium_HD_outs\M2M6\binned_outputs\square_002um"
path_output = r"X:\roy\viziumHD\analysis\python\version_11\organs\human_liver\output"
m2m6 = HiVis.new(path_image_fullres, path_input_data, path_output,
                          name="human_liver_M2M6",  properties=properties,
                        on_tissue_only=False,min_reads_in_spot=1, min_reads_gene=10)

#%% Crop M6

annotation_path = "X:\\roy\\other_projects\\Orans\\data_2025\\qupath\\export\\M6.geojson"
annotation_name = "patient"

mask_values = m2m6.add_annotations(annotation_path, annotation_name)
m6 = m2m6[m2m6["patient"] == "WholeTissue",:]
m6.rename("human_liver_M6", full=True)
# m6.export_images()

#%% Import single-cell

path = r"X:\oran\Data\Human_Liver_Project\Human_Liver_QuPath\visium_HD_analysis\VisiumHD_oran_liver_M6\export\human_liver_M2M6_subset_fullres_exp_6_no_classification_WITH_object_class_detections.csv"
segmentation = pd.read_csv(path,sep="\t")

obs2add = ["Nucleus/Cell area ratio",
           "Nucleus: Circularity",
           "Nucleus: Solidity",
           "Nucleus: Area µm^2",
           "Classification"]

m6.agg_cells(input_df=segmentation, name="SC",obs2add=obs2add)

geojson_path = r"X:\oran\Data\Human_Liver_Project\Human_Liver_QuPath\visium_HD_analysis\VisiumHD_oran_liver_M6\export\human_liver_M2M6_subset_fullres_exp_6_no_classification_WITH_object_class_cells.geojson"
m6.agg["SC"].import_geometry(geojson_path)

m6.agg["SC"].sync("Classification")

m6.agg["HEP"] = m6.agg["SC"][m6.agg["SC"]["Classification"] == "hepato-cell",:]
#%%% Add nucs to cells
import geopandas as gpd 

geojson_path = r"X:\oran\Data\Human_Liver_Project\Human_Liver_QuPath\visium_HD_analysis\VisiumHD_oran_liver_M6\export\human_liver_M2M6_subset_fullres_exp_6_no_classification_WITH_object_class_nucs.geojson"
gdf = gpd.read_file(geojson_path)
gdf["name"] = gdf["name"].str.replace("null__","")
m6.agg["nuc"] = m6.agg["SC"].copy()
name_to_id = pd.Series(gdf["id"].values,index=gdf["name"])
adata = m6.agg["nuc"].adata
common = adata.obs_names.intersection(name_to_id.index)
adata_sub = adata[common].copy()
new_idx = pd.Index(adata_sub.obs_names.map(name_to_id), name="cell_id")
adata_sub.obs_names = new_idx

m6["nuc_ID"] = m6["Cell_ID_SC"]
m6["nuc_ID"] = m6.adata.obs["nuc_ID"].map(name_to_id)
adata_sub.obs.index.name = "nuc_ID"
m6.agg["nuc"].adata = adata_sub
m6.agg["nuc"].import_geometry(geojson_path,object_type="detection")

#%% Plot cells and nucs
xlim = [400,700]
ylim = [2085,2235]
alpha = 0.3

fig, axes = plt.subplots(2,1,figsize=(7,7))
ax = m6.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Human liver",scalebar={"text_offset":0.06})

m6.agg["HEP"].adata.obs["temp"] = True
ax = m6.agg["HEP"].plot.cells("temp",xlim=xlim,ylim=ylim,line_color="black",image=True,ax=axes[1],
                             cmap=["orange","orange"],legend=False,alpha=alpha,scalebar={"text":False})
ax = m6.agg["nuc"].plot.cells("Classification",xlim=xlim,ylim=ylim,line_color="black",cmap=["red","cyan"],
                         title="",ax=ax,image=False,legend=False,alpha=alpha,scalebar=False)
bv_patch = mpatches.Patch(color='orange', label='Cytoplasm')
bv_patch2 = mpatches.Patch(color='cyan', label='Nucleus')
bv_patch3 = mpatches.Patch(color='red', label='NPCs')

ax.legend(handles=[bv_patch,bv_patch2,bv_patch3], title=None, loc='upper right')
ax.set_title(None)
plt.tight_layout()
m6.plot.save("cells_and_nuclei_CELLS",fig=fig)

#%% Zonation
#%%% Calculate Zonation score

markers_PC = ["CYP3A4", "ADH1B", "CYP1A2", "CYP2E1", "APOA2", "APOC1", "ADH4", "ADH1A", "APOH", "AMBP", 
              "GSTA2", "ADH1C", "SLCO1B3", "AOX1", "APOA5", "DCXR", "RBP4", "OAT", "CYP2C19", "GC"]
markers_PP = ["SERPINA1", "APOA1", "ALB", "C7", "NNMT", "HAMP", "ALDOB", "ASS1", "CYP2A7", "MGP", "A2M", 
              "FXYD2", "CCL21","HAL", "IGFBP2", "SDS", "AQP1", "CYP2A6", "FBLN1", "PTGDS"]

markers_PC = [g.upper() for g in markers_PC if g.upper() in m6.agg["SC"].adata.var_names]
markers_PP = [g.upper() for g in markers_PP if g.upper() in m6.agg["SC"].adata.var_names]
sum_pp = m6.agg["HEP"].adata[:,markers_PP].X.sum(axis=1)
sum_pc = m6.agg["HEP"].adata[:,markers_PC].X.sum(axis=1)
m6.agg["HEP"].adata.obs["eta"] = sum_pp / (sum_pp + sum_pc)
m6.agg["HEP"].plot.spatial("eta",cmap="viridis",image=False,save=True,size=5)
m6.agg["HEP"].plot.hist("eta",save=True,xlab="Eta score",bins=20)
plt.show()
m6.agg["HEP"] = m6.agg["HEP"][~m6.agg["HEP"].adata.obs["eta"].isna(),:]

#%%% Bin zones
num_bins = 4
eta_values = m6.agg["HEP"].adata.obs["eta"].values
breaks = jenks_breaks(eta_values, n_classes=num_bins)
zone_labels = range(1, num_bins + 1)
# m6.agg["HEP"].adata.obs["zone_non_smooth"] = pd.cut(eta_values, bins=breaks, labels=zone_labels,include_lowest=True)
m6.agg["HEP"].adata.obs["zone_non_smooth"] = pd.qcut(eta_values,q=num_bins,labels=zone_labels)


fig, axes = plt.subplots(1,3,figsize=(21,7))
m6.agg["HEP"].plot.hist("zone_non_smooth",ax=axes[0],title="Fisher-jenks binning")
m6.agg["HEP"].plot.spatial("zone_non_smooth",ax=axes[1],axis_labels=False,
    cmap="viridis",image=False, legend=False,size=5,title="Fisher-jenks binning")
m6.agg["HEP"].plot.hist("eta", ax=axes[2], xlab="Eta score",title="Fisher-jenks binning")
for b in breaks:
    axes[2].axvline(x=b, color='red', linestyle='--')
plt.tight_layout()
m6.agg["HEP"].plot.save("zonation_binning",fig=fig)

#%%% Smooth the zones
radius = 50
method = "median"
m6.agg["HEP"].adata.obs["zone_non_smooth"] = m6.agg["HEP"].adata.obs["zone_non_smooth"].astype(np.int8)
_ = m6.agg["HEP"].analysis.smooth("zone_non_smooth",radius=radius,method=method,new_col_name="zone")
m6.agg["HEP"]["zone"] = round(m6.agg["HEP"].adata.obs["zone"])
m6.agg["HEP"].sync("zone")

# Matnorm
m6.agg["HEP"].adata.layers["matnorm"] = HiVis.HiVis_utils.matnorm(m6.agg["HEP"].adata.X,axis="row")
#%%% plot zones
fig, axes = plt.subplots(2, 1, figsize=(7,7))

ax = m6.plot.spatial(ax=axes[0],title="Healthy human liver",img_resolution="high",scalebar={"text_offset":0.06})
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="23%", pad=0.1, sharey=ax)  # size controls width
cax.set_visible(False) 

ax = m6.agg["HEP"].plot.spatial("zone",cmap="viridis",image=False,size=3,ax=axes[1],title="",
                           scalebar={"text":False},legend_title="")
ax.text(1.04,0.88,"Portal",transform=ax.transAxes,fontsize=14,color="black")
ax.text(1.04,0.07,"Central",transform=ax.transAxes,fontsize=14,color="black")
plt.tight_layout()

m6.agg["HEP"].plot.save("zonation_SPATIAL",fig=fig)

#%% Nuclear retention
#%%% Subset mid zones and remove mito genes
m6["retention"] = m6["InNuc"]
m6.update_meta("retention", {1:"nuc",0:"cyto"})

non_mito_genes = ~m6.adata.var.index.str.startswith("MT-")

mid = m6[m6.adata.obs["zone"].isin([2,3]),non_mito_genes]

#%%% DGE
dge = mid.analysis.dge("retention",group1="nuc",group2="cyto",method="fisher_exact",two_sided=False)
m6.plot.save("DGE_retention_midzones",fig=dge)

# dge = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_liver\output\human_liver_M6_DGE_retention_midzones.csv",index_col=0)

#%%% Plot MA (new)
qval_thresh = 0.05
exp_thresh = 1e-5

nuc_examples = ["MST1","CYP3A5","MLXIPL","PAH","PLGLB2"]
cyto_examples = ["APOE","APOA1","ALB","FTL","FABP1"]

plot = dge.loc[dge["expression_mean"] >= exp_thresh].copy()

plot["expression_mean"] = np.log10(plot["expression_mean"])

nuc_plot = plot.loc[plot["log2fc"] >= 0]
cyto_plot = plot.loc[plot["log2fc"] < 0]

nuc_genes = nuc_plot.index[(nuc_plot["qval"] < qval_thresh)]
nuc_genes_text = nuc_plot.index[(nuc_plot["qval"] < qval_thresh) & nuc_plot.index.isin(nuc_examples) ]


cyto_genes = cyto_plot.index[(cyto_plot["qval"] < qval_thresh)]
cyto_genes_text = cyto_plot.index[(cyto_plot["qval"] < qval_thresh) & cyto_plot.index.isin(cyto_examples) ]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", 
                                          "log2fc",genes=nuc_genes_text,color_genes="darkcyan",
                                          genes2=cyto_genes_text,color_genes2="darkorange",repel=1,
                                          text=True, color="gray",y_line=0,figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", "log2fc",
                                          genes=nuc_genes,color_genes="darkcyan",
                                          genes2=cyto_genes,color_genes2="darkorange",
                                          text=False, color="gray",ax=ax,
                                          xlab="log10(expression)",ylab="log2(nucleus/cytoplasm)",
                                          title="Cytoplasmic bias - human liver")

ax.text(0.98, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="darkcyan",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="darkorange",
        bbox=dict(facecolor='white', edgecolor='black'))


plt.savefig(r"X:\roy\viziumHD\analysis\python\version_11\organs\human_liver\output\human_liver_M6_MA_retention.pdf")
plt.savefig(r"X:\roy\viziumHD\analysis\python\version_11\organs\human_liver\output\human_liver_M6_MA_retention.png")

#%% Save

m6.save()

