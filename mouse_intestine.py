# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:29:19 2025

@author: royno
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
from tqdm import tqdm
from scipy.stats import spearmanr
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
from HiVis import HiVis

#%% Import data

properties = {"organism":"mouse",
              "organ":"intestine",
              "sample_id":"mouse_intestine",
              "source":"10X"}

# path_input = directory + '/input'
path_image_fullres = r"X:\roy\viziumHD\data\intestine\mouse\data\Visium_HD_Mouse_Small_Intestine_tissue_image.btf"
path_input_data = r"X:\roy\viziumHD\data\intestine\mouse\data\square_002um"
path_output = r"X:\roy\viziumHD\analysis\python\version_11\organs\mouse_intestine\output"

si = HiVis.new(path_image_fullres, path_input_data, path_output,
                          name="mouse_intestine",  properties=properties,
                          on_tissue_only=False,min_reads_in_spot=1, min_reads_gene=10)
# si = HiVis.load(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\mouse_intestine.pkl")


si.plot.spatial(save=True,img_resolution="high")
#%% Add annotations

condition_name = "apicome_manual"
annotations_path = r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - apicome manual.geojson"
si.add_annotations(annotations_path,condition_name)
si.plot.spatial(condition_name,legend=False,cmap="Set1",xlim=[0,400],ylim=[3300,3700],alpha=0.7)


condition_name = "villus"
annotations_path = r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - villi.geojson"
si.add_annotations(annotations_path,condition_name)
si.agg_from_annotations(condition_name+"_id",name="villi",geojson_path=annotations_path)
si.agg["villi"].plot.cells(condition_name,legend=False,cmap="tab20")
# Plot villi
sub = si.crop(xlim=[0,400],ylim=[3300,3700])
unique_ann = np.unique(sub.agg["villi"]["villus"])
colors = ["red","c","brown","c","green","blue","yellow","green"]
fig, ax = plt.subplots(1,1,figsize=(7,7))
for i, v in enumerate(unique_ann):
    sub.agg["temp"] = sub.agg["villi"][sub.agg["villi"]["villus"] == v,1:2]
    try:
        sub.agg["temp"].plot.cells("villus",line_color=colors[i],legend=False, alpha=0.2,img_resolution="high",
                                   cmap=[colors[i],colors[i]],ax=ax,image=True,linewidth=2,scalebar={"length":100,"text":False})
    except:
        pass
ax.set_title("Individual villi segmentation")
si.agg["villi"].plot.save("villi_annotations",ax=ax)


condition_name = "tissue_classifier"
annotations_path = r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - tissue_classifier_prediction.tif"
si.add_mask(annotations_path,condition_name)
mask_names_tissue = {0:"immune", 1:"muscle", 2:"tissue", 3:"lumen"}
si.update_meta(condition_name, mask_names_tissue)

si["temp"] = si.adata.obs["tissue_classifier"].str.capitalize()
si.update_meta("temp",{"Tissue":"Other","lumen":np.nan})

ax = si.plot.spatial("temp",xlim=[0,400],ylim=[3300,3700],legend="upper right",exact=True,scalebar={"text":False},
                title="Muscle classifier",cmap=["red","gray"],alpha=0.4,legend_title="",img_resolution="full")
  
si.plot.save("tissue_classifier_SPATIAL",dpi=1200)


condition_name = "apicome_classifier"
annotations_path = r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - apicome classifier training_prediction.tif"
si.add_mask(annotations_path,condition_name)
si.plot.spatial("apicome_classifier",xlim=[0,400],ylim=[3300,3700],cmap="Set1")

mask_names_tissue = {0:"stroma", 1:"apical", 2:"basal", 3:np.nan,4:"nucleus"}
si.update_meta(condition_name, mask_names_tissue)

condition_name = "region"
annotations_path = r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - region to analyse.geojson"
si.add_annotations(annotations_path,condition_name)
si.agg_from_annotations(condition_name+"_id",name="region",geojson_path=annotations_path)
si.agg["region"].plot.cells(condition_name,legend=False,cmap=["c","c"],img_resolution="low",
                            alpha=0.2,title="Selected ROI for zonation analysis",save=True)

#%% Assign zonation
spots = si.adata.obs.copy()
spots["barcode"] = spots.index.values
spots['dist_from_muscle'] = np.nan
spots['villi_spots_count'] = np.nan
spots['muscle_median_x'] = np.nan
spots['muscle_median_y'] = np.nan

spots['villus_id'] = spots['villus_id'].fillna(-1)

spots.reset_index(inplace=True,drop=True)


def process_group(group):
    muscle_spots = group[group['tissue_classifier'] == 'muscle']
    if group.name != -1:
        tissue_spots = group[group['tissue_classifier'] == 'tissue']
        muscle_median_x, muscle_median_y = muscle_spots[['pxl_col_in_fullres', 'pxl_row_in_fullres']].median()
        villus_median_x, villus_median_y = tissue_spots[['pxl_col_in_fullres', 'pxl_row_in_fullres']].median()
        group['muscle_median_x'] = muscle_median_x
        group['muscle_median_y'] = muscle_median_y
        group.loc[group['tissue_classifier'] == 'tissue', 'dist_from_muscle'] = \
            np.sqrt((group['pxl_row_in_fullres'] - muscle_median_y)**2 + \
                    (group['pxl_col_in_fullres'] - muscle_median_x)**2)
    return group



tqdm.pandas(desc="Calculating villi/intestinal axes score")
spots = spots.groupby('villus_id').progress_apply(process_group)
spots['villus_id'] = spots['villus_id'].replace(-1, np.nan)


#%%% bin
n_bins = 30
data = spots['dist_from_muscle'][np.isfinite(spots['dist_from_muscle'])]
counts, bins = np.histogram(data, bins=n_bins)

zones_values = {
    "Tip": (900, bins.max()),
    "Mid": (650, 900),
    "Base": (400, 650),
    "Crypt": (200, 400)
}

def plot_zone_hist(data, ax=None, n_bins=30,height_factor=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    counts, bins, patches = ax.hist(data, bins=n_bins, edgecolor=None)
    norm = plt.Normalize(bins.min(), bins.max())
    for count, bin, patch in zip(counts, bins, patches):
        color = plt.cm.viridis(norm(bin))
        patch.set_facecolor(color)

    def add_annotation(text, bin_range):
        bin_indices = [i for i in range(len(bins)-1) if bins[i] >= bin_range[0] and bins[i+1] <= bin_range[1]]
        left = bins[bin_indices[0]]
        right = bins[bin_indices[-1] + 1]
        height = height_factor * max(counts)
        ax.plot([left, left, right, right], [height, height + 0.02, height + 0.02, height], 'k-', lw=1.5)
        ax.text((left + right) / 2, height + 1000, text, ha='center', va='bottom', fontsize=10)


    for label, (low, high) in zones_values.items():
        add_annotation(label, (low, high))
    ax.set_title(None)
    ax.set_xlabel('Distance from muscle (pixels)')
    ax.set_ylabel('Number of spots')
    return ax

plot_zone_hist(spots['dist_from_muscle'], n_bins=n_bins)    

def map_zones(distance_from_muscle):
    if distance_from_muscle > zones_values["Tip"][0]:
        return "Tip"
    elif distance_from_muscle > zones_values["Mid"][0]:
        return "Mid"
    elif distance_from_muscle > zones_values["Base"][0]:
        return "Base"
    elif distance_from_muscle > zones_values["Crypt"][0]:
        return "Crypt"
    else:
        return np.nan
        
tqdm.pandas(desc="Assagning zones")
spots['zone'] = spots.progress_apply(lambda row: map_zones(row['dist_from_muscle']), axis=1, result_type='expand')

spots.set_index('barcode', inplace=True)

si["zone"] = spots["zone"]
si["dist_from_muscle"] = spots["dist_from_muscle"]

si.update_meta("tissue_classifier",{"lumen":np.nan})



#%% Single cell 

def _count_apical(series):
        return (series == 'apical').sum().astype(float)

def _count_basal(series):
    return (series == 'basal').sum().astype(float)

obs2add = ["Cell: Circularity",
           "Cell: Solidity",
           "Nucleus/Cell area ratio",
           "Nucleus: Circularity",
           "Nucleus: Solidity",
           "Nucleus: Area µm^2",
           "ROI: 2.00 µm per pixel: Hematoxylin: Mean",
           "ROI: 2.00 µm per pixel: Hematoxylin: Std.dev",
           "ROI: 2.00 µm per pixel: Hematoxylin: Max",
           "ROI: 2.00 µm per pixel: Hematoxylin: Min",
           "ROI: 2.00 µm per pixel: Eosin: Mean",
           "ROI: 2.00 µm per pixel: Eosin: Std.dev",
           "ROI: 2.00 µm per pixel: Eosin: Max",
           "ROI: 2.00 µm per pixel: Eosin: Min"]

si.agg_cells(input_df=segmentation, name="SC",geojson_path=geojson_path,obs2add=obs2add,
                obs2agg={"dist_from_muscle":np.mean,"zone":None,"tissue_classifier":None,
                         "villus_id":None,"apicome_manual":[_count_apical,_count_basal],"region":None})

adatas = {}


si.agg["SC"].plot.cells(save=True,xlim=[0,400],ylim=[3300,3700],title="Mouse jejunum",
                        scalebar={"text":False,"color":"black"})
#%%% UMAP
# sc_to_use = "SC"
# for sc_to_use in ["SC","SC2","analysis2"]:
si.agg["SC_sub"] = si.agg["SC"][si.agg["SC"]["region"] == "positive",:]
for sc_to_use in ["SC_sub"]:

    umi_thresh = 0
    gene_count_thresh = 3
    
    high_exp_cells = si.agg[sc_to_use]["nUMI"] >= umi_thresh
    si.agg[sc_to_use+"_"] = si.agg[sc_to_use][high_exp_cells,:]
    sc.pp.filter_genes(si.agg[sc_to_use+"_"].adata, min_cells=gene_count_thresh)
    
    adata_sc = si.agg[sc_to_use+"_"].adata.copy()
    
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    adata_sc.var["expression_mean"] = np.array(adata_sc.X.mean(axis=0)).flatten()/1e4
    adata_sc.var["expression_max"] =  np.array(adata_sc.X.max(axis=0).toarray()).flatten()/1e4
    sc.pp.log1p(adata_sc)
    
    adata_sc.layers["log_norm"] = adata_sc.X.copy()
    si.agg[sc_to_use+"_"].merge(adata_sc,layer="log_norm")
    
    adata_sc.raw = adata_sc
    
    sc.pp.highly_variable_genes(adata_sc, n_top_genes=3000)
    adata_sc = adata_sc[:,(adata_sc.var['highly_variable'] == True) ].copy()
    sc.pp.scale(adata_sc, max_value=10)
    sc.tl.pca(adata_sc, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata_sc, log=True)
    sc.pp.neighbors(adata_sc, n_neighbors=15, n_pcs=30) 
    sc.tl.umap(adata_sc)
    
    resolution = 0.5
    sc.tl.leiden(adata_sc,resolution=resolution,random_state=0)
    # sc.pl.umap(adata_sc,color=["leiden","nUMI","dist_from_muscle","tissue_classifier"],size=3,title=sc_to_use)
    sc.pl.umap(adata_sc,color=["leiden"],size=3,title=sc_to_use)
    # rank genes
    sc.tl.rank_genes_groups(adata_sc, groupby="leiden", method="wilcoxon", use_raw=True,
                            n_genes=100, corr_method="benjamini-hochberg", key_added="rgg")
    
    # get top3 markers per cluster
    rgg_df = sc.get.rank_genes_groups_df(adata_sc, group=None,key="rgg")
    rgg_df = rgg_df[(rgg_df["pvals_adj"] < 0.05) & (rgg_df["logfoldchanges"] > 0)]
    top3 = rgg_df.sort_values(["group","pvals_adj","logfoldchanges"], ascending=[True,True,False]).groupby("group").head(3)
    
    # quick dotplot of top 3 per cluster
    sc.pl.rank_genes_groups_dotplot(adata_sc, key="rgg", n_genes=3, groupby="leiden", swap_axes=True)
    
    # or exact genes you selected
    gene_map = {g: sub["names"].tolist() for g, sub in top3.groupby("group")}
    sc.pl.dotplot(adata_sc, var_names=gene_map, groupby="leiden", standard_scale="var", swap_axes=True)
    
    adatas[sc_to_use] = adata_sc
    si.agg[sc_to_use+"_"].merge(adata_sc,obs="leiden",var=list(adata_sc.var.columns))
    
    plt.show()
raise StopIteration("***")
#%%% Plot dotplot markers of all clusters
sc_to_use = "SC_sub"

genes = ['Igha','Lgals2','Defa21','Ighm','Tpm2','Anpep','Ccl5','Lct','Sct','Ada','Hck','Krt19','Epcam']#analysis2 umi-thresh=100
genes = ['Epcam','Anpep','Igha','Tpm2','Krt19','Defa21','Ada',"Ighg1"] #SC umi-thresh=0

dp  = sc.pl.dotplot(adatas[sc_to_use], var_names=genes, groupby="leiden",show=False,
                   standard_scale="var", swap_axes=False,title=None,figsize=(5,5))
dp["mainplot_ax"].tick_params(axis='x', rotation=45) 
dp["mainplot_ax"].set_ylabel("Cluster")
fig = plt.gcf()

si.agg[sc_to_use+"_"].plot.save("DOTPLOT",fig=fig)

#%%% Assign clusters

leiden_map = {"0": "Enterocyte mid", "1": "Plasma", "5": "Paneth", "2": "Muscle", "3": "Enterocyte base","4": "Enterocyte base",
 "6": "Enterocyte tip", "7": "Plasma"}
leiden_map_broad = {"0": "Epithel", "1": "Immune", "2": "Muscle", "3": "Epithel", "4": "Epithel", "5": "Epithel",
 "6": "Epithel", "7": "Immune"}

adata_sc.obs["cell_type"] = adata_sc.obs["leiden"].map(leiden_map)
adata_sc.obs["cell_type_broad"] = adata_sc.obs["leiden"].map(leiden_map_broad)

# sc.pl.umap(adata_sc,color=["leiden","cell_type","cell_type_broad"],size=3)   
si.agg[sc_to_use+"_"].merge(adata_sc,obs=["leiden","cell_type","cell_type_broad"])

#%%% Plot dotplot celltypes
genes = ['Igha','Lgals2','Defa21','Ighm','Tpm2','Anpep','Ccl5','Lct','Sct','Ada','Hck','Krt19','Epcam']#analysis2 umi-thresh=100
genes = ['Epcam','Lgals2','Anpep','Ada','Tpm2','Cd74','Defa21','Igha']
genes = ['Lgals2','Krt19','Anpep','Apoa1','Epcam','Ada','Tpm2','Acta2','Cd74','Clu','Defa21','Lyz1','Igha','Jchain']
genes = ['Lgals2','Krt19','Anpep','Apoa1','Epcam','Ada','Tpm2','Acta2','Defa21','Lyz1','Igha','Jchain']

dp = sc.pl.dotplot(adatas[sc_to_use], var_names=genes, groupby="cell_type",show=False, standard_scale="var", swap_axes=True,title=None,figsize=(5,5))
for label in dp["mainplot_ax"].get_xticklabels():
    label.set_rotation(35)
    label.set_ha("right") 
dp['mainplot_ax'].set_title(" ")
fig = plt.gcf()

si.agg[sc_to_use+"_"].plot.save("DOTPLOT_assigned",fig=fig)

#%%% Plot UMAPs and spatial
#%%%% celltypes UMAP
cell_type_cmap = {"Plasma":"orange","B":"gold","Other":"gray",
                  "Enterocyte mid":"dodgerblue","Enterocyte tip":"lightblue","Enterocyte base":"cyan",
                  "Paneth":"lightgreen","Muscle":"Red"}
arr = si.agg[sc_to_use+"_"]["cell_type"]
si.agg[sc_to_use+"_"]["temp"] = np.where(arr == "Other", np.nan, arr)

ax = si.agg[sc_to_use+"_"].plot.umap("cell_type",size=10, title="Cell types",texts=True,
                                     figsize=(7,7),legend=False,cmap=cell_type_cmap)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
si.agg[sc_to_use+"_"].plot.save("UMAP_clustered",ax=ax,dpi=1200)

#%%%% Celltypes spatial

ax = si.agg[sc_to_use+"_"].plot.cells("temp",xlim=[0,400],ylim=[3300,3700],cmap=cell_type_cmap,figsize=(7,7),legend_title="",
                                      alpha=1,scalebar={"text":False},image=False,title="Cell types",legend="upper right",img_resolution="high")
leg = ax.get_legend()
if leg is not None:
    frame = leg.get_frame()
    frame.set_facecolor("white")  
    frame.set_edgecolor("black")  
    frame.set_alpha(1) 
si.agg[sc_to_use+"_"].plot.save("cell_type_SPATIAL",ax=ax)

#%%%% Broad spatial

cell_type_broad_cmap = {"Epithel":"dodgerblue","Immune":"orange","Other":"gray","Muscle":"red"}
arr = si.agg[sc_to_use+"_"]["cell_type_broad"]
si.agg[sc_to_use+"_"]["temp"] = np.where(arr == "Other", np.nan, arr)
# si.agg[sc_to_use+"_"]["temp"] = np.where(arr == "Plasma", "Immune", arr)

ax = si.agg[sc_to_use+"_"].plot.cells("temp",xlim=[0,400],ylim=[3300,3700],cmap=cell_type_broad_cmap,title="Cell types",
                                      image=False,alpha=1,scalebar={"text":False},figsize=(7,7),legend_title="",legend="upper right")
leg = ax.get_legend()
if leg is not None:
    frame = leg.get_frame()
    frame.set_facecolor("white")  
    frame.set_edgecolor("black")  
    frame.set_alpha(1) 

si.agg[sc_to_use+"_"].plot.save("cell_type_broad_SPATIAL",ax=ax)

#%%%% leiden UMAP broad
# arr = si.agg[sc_to_use+"_"]["cell_type_broad"]
# si.agg[sc_to_use+"_"]["temp"] = np.where(arr == "Other", "Immune", arr)
# arr = si.agg[sc_to_use+"_"]["cell_type_broad"]
# si.agg[sc_to_use+"_"]["temp"] = np.where(arr == "Plasma", "Immune", arr)

ax = si.agg[sc_to_use+"_"].plot.umap("cell_type_broad",size=10, title="Cell types",texts=True,
                                     figsize=(7,7),legend=False,cmap=cell_type_broad_cmap)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
# for r in  ax.collections:
#     r.set_rasterized(True)
si.agg[sc_to_use+"_"].plot.save("UMAP_broad_clustered",ax=ax,dpi=1200)

#%%%% leiden UMAP
leiden_cmap = {"1":"orange","7":"gray",
                  "0":"dodgerblue","6":"lightblue","4":"cyan",
                  "5":"lightgreen","2":"Red","3":"darkturquoise"}
ax = si.agg[sc_to_use+"_"].plot.umap("leiden",size=10, title="Leiden",texts=True,
                                     figsize=(7,7),legend=False,cmap=leiden_cmap)
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
# for r in  ax.collections:
#     r.set_rasterized(True)
si.agg[sc_to_use+"_"].plot.save("UMAP_leiden",ax=ax,dpi=1200)

# get_umap_df(si.agg[sc_to_use+"_"].adata,["leiden","cell_type", "cell_type_broad"],"SI_umaps")


#%% Zonation comparison to scRNAseq
#%%% plot zonation score

si.agg["SC"]["temp"] = si.agg["SC"]["dist_from_muscle"] * si.json["microns_per_pixel"]

ax = si.agg["SC"].plot.cells("temp",xlim=[0,400],ylim=[3300,3700],legend=False,image=False,
                          title="Distance from muscle",cmap="viridis",alpha=0.8,
                          legend_title="Distance (um)",scalebar={"text":False})
si.agg["SC"]["temp"] = si.agg["SC"].adata.obs["tissue_classifier"].str.capitalize()
si.agg["SC"].update_meta("temp",{"Tissue":np.nan,"Immune":np.nan})
ax = si.agg["SC"].plot.cells("temp",xlim=[0,400],ylim=[3300,3700],image=False,
                          title="Distance from muscle",cmap=["red","red"],alpha=1,
                          ax=ax,legend="upper right",legend_title="",scalebar=False)
si.agg["SC"].plot.save("ZONATION_SPATIAL",ax=ax)

si.agg["SC"]["temp"] = si.agg["SC"]["dist_from_muscle"] * si.json["microns_per_pixel"]

ax = si.agg["SC"].plot.cells("temp",xlim=[0,400],ylim=[3300,3700],
                          title="Distance from muscle",cmap="viridis",alpha=0.7,
                          legend_title="Distance (µm)")
fig = ax.get_figure()
cbar_ax = [a for a in fig.axes if a is not ax][0]
fig.savefig(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\mouse_intestine_SC\ZONATION_colorbar.pdf",
            bbox_inches=cbar_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()))

#%%% plot zones

si.agg["SC"].adata.obs["zone"] = pd.Categorical(si.agg["SC"].adata.obs["zone"], ordered=True,
                                                categories=["Crypt","Base","Mid","Tip"] )
ax = si.agg["SC"].plot.cells("zone",xlim=[0,400],ylim=[3300,3700],legend="upper right",img_resolution="high",alpha=0.8,
                             scalebar={"text":False},title="Cells zonation",legend_title="",image=False)
si.agg["SC"].plot.save("zone",ax=ax)

# Epithelial cells only
si.agg["SC_sub_"].adata.obs["zone_epi"] = si.agg["SC_sub_"].adata.obs["zone"]
si.agg["SC_sub_"].adata.obs.loc[si.agg["SC_sub_"]["cell_type_broad"]!="Epithel","zone_epi"] = np.nan
si.agg["SC_sub_"].adata.obs["zone_epi"] = pd.Categorical(si.agg["SC_sub_"].adata.obs["zone_epi"], ordered=True,
                                                categories=["Crypt","Base","Mid","Tip"] )

ax = si.agg["SC_sub_"].plot.cells("zone_epi",xlim=[0,400],ylim=[3300,3700],legend="upper right",img_resolution="full",
                             scalebar={"text":False},title="Cells zonation",legend_title="",image=True)

si.agg["SC"].plot.save("ZONE_EPI",ax=ax)

#%%% Epi genes
expression = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\input\Bahar2023.csv",index_col=0)
expression_telo = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\input\Bahar2020_telocytes.csv",index_col=0) 
expression["Fibroblasts"] = HiVis.HiVis_utils.matnorm(expression_telo["mean"].reindex(expression.index).fillna(0))
expression_ec = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\input\intestine_EC_pseudobulk.csv",index_col=0) 
expression_ec["mean"] = expression_ec.mean(axis=1)
expression["Endothel"] = HiVis.HiVis_utils.matnorm(expression_ec["mean"].reindex(expression.index).fillna(0))

fc_thresh = 1
epi_cols = ["VillusBottom","VillusTop","Goblet"]
epi = expression[epi_cols].max(axis=1)
other = expression.columns[~expression.columns.isin(epi_cols)]
other = expression[other].max(axis=1)
pn = epi[epi>0].min()
ratio = (epi+pn) / (other+pn)
epi_genes = expression.index[ratio >= fc_thresh]

si_epi = si[si["region"]=="positive",si.adata.var.index.str.upper().isin(epi_genes.str.upper())]

#%%% DGE
# met = "bins"
# met = "sc all"
met = "sc epi"

if met == "bins":
    zonation = si_epi.analysis.dge("zone", group1="Tip", group2="Base",method="fisher_exact") 
elif met == "sc all":
    zonation = si_epi.agg["SC_sub_"].analysis.dge("zone", group1="Tip", group2="Base",method="wilcox") 
elif met == "sc epi":
    sub = si_epi.agg["SC_sub_"][si_epi.agg["SC_sub_"]["cell_type_broad"]=="Epithel",:]
    zonation = sub.analysis.dge("zone", group1="Tip", group2="Base",method="wilcox") 

zonation["gene"] = zonation.index
zonation.to_csv(f"{si_epi.path_output}/zonation_epi_genes.csv")

zonation['gene'] = zonation['gene'].str.upper()

#%%% import previous data - Andreases scRNAseq

inna_zonation = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\input\mouse_intestines_sc_innas.csv")
inna_zonation = inna_zonation.filter(regex='^enterocyte_V|gene')
inna_zonation['Base'] = (inna_zonation['enterocyte_V1'] + inna_zonation['enterocyte_V2'])/2
inna_zonation['Tip'] = (inna_zonation['enterocyte_V5'] + inna_zonation['enterocyte_V6'])/2
pn = min([inna_zonation.loc[inna_zonation['Tip'] > 0, 'Tip'].min(),
          inna_zonation.loc[inna_zonation['Base'] > 0, 'Base'].min()])
inna_zonation['ratio'] = (inna_zonation['Tip'] + pn) / (inna_zonation['Base'] + pn) 
inna_zonation['log2fc'] = np.log2(inna_zonation['ratio'])
inna_zonation['expression_max'] = np.max(inna_zonation[['Tip', 'Base']], axis=1)

columns = ["gene","Tip","Base","log2fc","expression_max"]
zonation_merged = pd.merge(zonation[columns],inna_zonation[columns],on="gene",suffixes=["_viziumHD","_inna"],how="inner")
#%%% plot correlation
exp_thresh = 1e-4

plot = zonation_merged[(zonation_merged["expression_max_inna"] >= exp_thresh) &
                       (zonation_merged["expression_max_viziumHD"] >= exp_thresh)]

signif_genes=["ADA","NT5E","REG3A","KRT19","APOA4","ENPP3","PMP22","MS4A18","TFRC"]
# signif_genes=plot["gene"].tolist()

ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_inna", "log2fc_viziumHD",genes=signif_genes,
                                          repel=True,color_genes="black",figsize=(6,6),
                                          title="Correlation with Moor et al (2018)",
                         xlab="log2(tip/base) - scRNA-seq",ylab="log2(tip/base) - VisiumHD")

corr, pval = spearmanr(plot['log2fc_inna'],  plot['log2fc_viziumHD'])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95, f"r = {corr:.2f}, p = {pval:.2g}", transform=plt.gca().transAxes,color="black")

si.agg["SC"].plot.save("zonation_LCM_correlation_epi_genes",ax=ax)   

#%% Apicome

# Define conditions. in base/mid of villi, in proximal jejunum
base_mid = si.adata.obs["zone"].isin(["Base", "Mid"]) & ~si.adata.obs["villus"].isna()

si.adata.obs["apicome_classifier_base"] = None
si.adata.obs.loc[base_mid, "apicome_classifier_base"] = si.adata.obs["apicome_classifier"]
si.adata.obs["apicome_classifier_region"] = None
region = si["region"] == "positive"
si.adata.obs.loc[base_mid & region, "apicome_classifier_region"] = si.adata.obs["apicome_classifier"]

apicome = "apicome_classifier_region"

#%%% Plot apicome spatial
# import matplotlib.patches as mpatches

si["temp"] = si.adata.obs[apicome].str.capitalize()
si.adata.obs.loc[si["temp"] == None,"temp"] = np.nan
# si.adata.obs.loc[si["temp"] == "Stroma","temp"] = np.nan
si.update_meta("temp",{"Stroma":np.nan,"Nucleus":np.nan})
si.adata.obs["temp"].unique()

apicome_cmap = {"Apical":"red","Basal":"blue","Nucleus":"orange"}


xlim = [50,250]
ylim = [3375,3475]
# xlim=[50,300];ylim=[3315,3515]

fig, axes = plt.subplots(2,1,figsize=(7,7))

ax = si.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Mouse intestine") 

ax = si.plot.spatial("temp",xlim=xlim,ylim=ylim,title="Subcellular pixel classifier",
                     alpha=0.5,legend_title="",exact=True,ax=axes[1],
                     cmap=apicome_cmap,legend="upper left",scalebar={"text":False},)

# legend_patches = [mpatches.Patch(color=c, label=lbl) for lbl, c in apicome_cmap.items()]
# ax.legend( handles=legend_patches,loc="upper center",bbox_to_anchor=(0.5, 0),  ncol=len(apicome_cmap),frameon=True)
plt.tight_layout()
si.plot.save("apicome_SPATIAL",ax=ax)


#%%% filter epithel specific genes
lcm = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\input\roy_apicome_medians.csv")
lcm.rename(columns={"log2_fc":"log2fc"},inplace=True)

columns = ["gene","expression_mean","qval","log2fc"]
si_epi = si[:,si.adata.var.index.str.upper().isin(epi_genes.str.upper())]

#%%% DGE
# apicomes = ["apicome_manual","apicome_classifier","apicome_classifier_base","apicome_classifier_region"]
df = si_epi.analysis.dge(apicome, group1="apical", group2="basal",method="fisher_exact",two_sided=False) 
df["gene"] = df.index
df.to_csv(f"{si_epi.path_output}/{apicome}_epi_genes.csv")

#%%% plot MA

qval_thresh = 0.05
exp_thresh = 1e-5
exp_thresh_extreme = 1e-80

apical_examples = ["Apob"]
basal_examples = ["Net1"]


plot = df.loc[df["expression_min"] >= exp_thresh].copy()
plot["expression_min"] = np.log10(plot["expression_min"])


apical_plot = plot.loc[plot["log2fc"] >= 0]
basal_plot = plot.loc[plot["log2fc"] < 0]

apical_genes = apical_plot.index[(apical_plot["qval"] < qval_thresh)]
apical_genes_text = apical_plot.index[(apical_plot["qval"] < exp_thresh_extreme) & ~apical_plot.index.isin(apical_examples) ]
basal_genes = basal_plot.index[(basal_plot["qval"] < qval_thresh)]
basal_genes_text = basal_plot.index[(basal_plot["qval"] < exp_thresh_extreme) & ~basal_plot.index.isin(basal_examples) ]


ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=apical_genes_text,color_genes="red",
                                          genes2=basal_genes_text,color_genes2="blue",repel=True,
                                          text=True, color="gray",figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=apical_genes,color_genes="red",
                                          genes2=basal_genes,color_genes2="blue",
                                          text=False, color="gray",ax=ax)


plot2 = plot.loc[plot["gene"].isin(apical_examples)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_min","log2fc",
                                          ax=ax,size=35,color_genes="red",repel=True,bold=True,
                                          genes=apical_examples)
plot2 = plot.loc[plot["gene"].isin(basal_examples)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_min","log2fc",
                                          xlab="log10(expression)", bold=True,
                                          title="Polarized genes - mouse intestine",ylab="log2(apical/basal)",
                                          ax=ax,size=35,color_genes="blue",repel=True,y_line=0,
                                          genes=basal_examples)
ax.text(0.98, 0.95,"Apical mRNAs",transform=ax.transAxes,ha='right',color="red",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Basal mRNAs",transform=ax.transAxes,ha='right',color="blue",
        bbox=dict(facecolor='white', edgecolor='black'))
si.plot.save(f"{apicome}_MA_epi_genes",ax=ax)  

#%%% Compare with LCM
#%%%% Plot correlation
exp_thresh = 1e-5
qval_thresh = 0.25

lcm_viz = pd.merge(df[columns],lcm[columns],on="gene",suffixes=("_viz","_lcm"))
lcm_viz = lcm_viz.loc[(lcm_viz["expression_mean_viz"] >= exp_thresh) & 
                      (lcm_viz["expression_mean_lcm"] >= exp_thresh),:]

fc_thresh = 0.5
lcm_viz = lcm_viz.loc[(abs(lcm_viz["log2fc_lcm"]) >= fc_thresh) & 
                      (abs(lcm_viz["log2fc_viz"]) >= fc_thresh),:]

lcm_viz = lcm_viz[lcm_viz["gene"].str.upper().isin(epi_genes.str.upper())]

signif_vis = (lcm_viz["qval_viz"] <= qval_thresh)
signif_lcm = (lcm_viz["qval_lcm"] <= qval_thresh)

signif_genes = lcm_viz.loc[signif_vis & signif_lcm,"gene"].tolist()
plot = lcm_viz.loc[lcm_viz["gene"].isin(signif_genes)]
 
genes = ["Pigr","Net1","Cyb5r3","Golga4","Gda","Mgat4a","Ncor1", "Asah2","Cdhr2","Apob","Enpep","Ace2"]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_viz", "log2fc_lcm",genes=genes,
                                          repel=True,color_genes="black",title=apicome,figsize=(6,6),
                         xlab="log2(apical/basal) - VisiumHD",ylab="log2(apical/basal) - LCM", 
                         x_line=0, y_line=0)
ax.set_title("Correlation with Novoselsky Roy et al. 2024")

corr, pval = spearmanr(plot['log2fc_viz'],  plot['log2fc_lcm'])
pval = pval if pval > 0 else 1e-300

ax.set_xlim([-2.7,1.5])
ax.set_ylim([-3.99,2.9])
ax.text(0.05, 0.95, f"r = {corr:.2f}, p = {pval:.2g}", transform=plt.gca().transAxes,color="black")

si.plot.save(f"{apicome}_LCM_correlation_epi_genes",ax=ax)   

#%%%% Venn diagram
from matplotlib_venn import venn2
qval_thresh_vis = 1e-5
signif_vis = set(lcm_viz.loc[lcm_viz["qval_viz"] <= qval_thresh_vis, "gene"])
signif_lcm = set(lcm_viz.loc[lcm_viz["qval_lcm"] <= qval_thresh, "gene"])
only_vis = signif_vis - signif_lcm
only_lcm = signif_lcm - signif_vis
both     = signif_vis & signif_lcm

fig, ax = plt.subplots(figsize=(3, 3))
venn = venn2([signif_vis, signif_lcm],set_labels=('VisiumHD', 'LCM'),ax=ax)

label_lcm = venn.set_labels[1]  # second label = "LCM"
x, y = label_lcm.get_position()
label_lcm.set_x(x - 0.06)   # move right
label_lcm.set_y(y + 0.62)
label_vis = venn.set_labels[0]  # second label = "LCM"
x, y = label_vis.get_position()
label_vis.set_x(x + 0.23)   # move right
label_vis.set_y(y + 1.05)

label = f"$q_\\mathrm{{vis}} = 10^{{{int(f'{qval_thresh_vis:.0e}'.split('e')[1])}}}$"
ax.set_title(label, y=0.1)
plt.tight_layout()
si.plot.save(f"{apicome}_LCM_VENN_q_{qval_thresh_vis}",ax=ax)   

#%%% Compare tip/base apicome

tip = si_epi[si_epi.adata.obs["zone"].isin(["Tip"]) & (si_epi["region"] == "positive"),:]
base = si_epi[si_epi.adata.obs["zone"].isin(["Base"]) & (si_epi["region"] == "positive"),:]

df_tip = tip.analysis.dge("apicome_classifier", group1="apical", group2="basal",method="fisher_exact") 
df_base = base.analysis.dge("apicome_classifier", group1="apical", group2="basal",method="fisher_exact") 

tip_base = df_tip.merge(df_base, on="gene", suffixes=('_tip', '_base'))

#%%%% Plot comparison
qval_thresh = 0.05
exp_thresh = 1e-5

signif_genes = tip_base.loc[(tip_base["qval_base"] <= qval_thresh) & (tip_base["qval_tip"] <= qval_thresh) &
                            (tip_base["expression_mean_base"] >= exp_thresh) &  (tip_base["expression_mean_tip"] >= exp_thresh),"gene"].tolist()

plot = tip_base.loc[tip_base["gene"].isin(signif_genes)]
 
genes = ["Ace2","Pigr","Cyb5r3","Net1","Apob","Lct","Cars2","Drg1","Hook2"]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_base", "log2fc_tip",genes=genes,figsize=(6,6),
                                          repel=True,color_genes="black",title="Polarization across villi zones",
                         xlab="log2(apical/basal) - Base of villi",ylab="log2(apical/basal) - Tip of villi", 
                         x_line=0, y_line=0)


corr, pval = spearmanr(plot['log2fc_base'],  plot['log2fc_tip'])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95, f"r = {corr:.2f}, p = {pval:.2g}", transform=plt.gca().transAxes,color="black")

si.plot.save("tip_vs_base_apicome_sctter",ax=ax)   

#%% Nuclear retention
condition_name = "apicome_classifier_region"

viz = si_epi

# move the "nuc" classification from "apicome":
if not "nuc_cyto" in viz.adata.obs.columns: 
    viz.adata.obs["nuc_cyto"] = viz.adata.obs[condition_name].copy()
    viz.update_meta(name="nuc_cyto", values={"apical": "cyto", "basal": "cyto"})
#%%% plot retention spatial
xlim = [50,250]
ylim = [3375,3475]

fig, axes = plt.subplots(2,1,figsize=(7,7))
ax = viz.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Mouse intestine",scalebar={"text_offset":0.06}) #,scalebar={"text_offset":0.06}
viz["temp"] = np.where(np.isin(viz["nuc_cyto"], ["nucleus","cyto"]), viz["nuc_cyto"], np.nan)
viz["temp"] = viz.adata.obs["temp"].str.capitalize()
viz.update_meta("temp",{"Cyto":"Cytoplasm"})
ax = viz.plot.spatial("temp",xlim=xlim,ylim=ylim,legend="upper right",legend_title="",ax=axes[1],
                     title="",exact=True,alpha=0.4,cmap=["orange","cyan"],scalebar={"text":False})
plt.tight_layout()
viz.plot.save("retention_SPATIAL",fig=fig)
#%%% DGE
df = viz.analysis.dge("nuc_cyto", group1="nucleus", group2="cyto")

df = df[["qval","pval","log2fc","nucleus","cyto"]]
df["expression_mean"] = df[["nucleus","cyto"]].mean(axis=1)
df["gene"] = df.index
df.to_csv(f"{si_epi.path_output}/retention.csv")

# df = pd.read_csv(f"{path_output}/retention.csv",index_col=0)
#%%% plot MA (new)
qval_thresh = 0.05
exp_thresh = 1e-5


nuc_examples = ["Agrn","Naip6","Clec2h","Myo15b","Xdh","Slc7a15"]
cyto_examples = ["Apob","Anpep","Clca4b","Fabp2","Apoa4"]

plot = df.loc[df["expression_mean"] >= exp_thresh].copy()
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
                                          title="Cytoplasmic bias - mouse intestine")

ax.text(0.98, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="darkcyan",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="darkorange",
        bbox=dict(facecolor='white', edgecolor='black'))


plt.savefig(rf"{path_output}/mouse_intestine_nuc_retention_MA.pdf")
plt.savefig(rf"{path_output}/mouse_intestine_nuc_retention_MA.png")
# si.plot.save("nuc_retention_MA",ax=ax)  


#%%% Compare tip/base retention

tip = si_epi[si_epi.adata.obs["zone"].isin(["Tip"]) & (si_epi["region"] == "positive"),:]
base = si_epi[si_epi.adata.obs["zone"].isin(["Base"]) & (si_epi["region"] == "positive"),:]
for viz in [tip,base]:
    viz["nuc_cyto"] = viz.adata.obs["apicome_classifier"].copy()
    viz.update_meta(name="nuc_cyto", values={"apical": "cyto", "basal": "cyto"})
    
df_tip = tip.analysis.dge("nuc_cyto", group1="nucleus", group2="cyto",method="fisher_exact") 
df_base = base.analysis.dge("nuc_cyto", group1="nucleus", group2="cyto",method="fisher_exact") 

tip_base = df_tip.merge(df_base, on="gene", suffixes=('_tip', '_base'))

#%%% Plot comparison
qval_thresh = 0.1
exp_thresh = 1e-5

signif_genes = tip_base.loc[(tip_base["qval_base"] <= qval_thresh) & (tip_base["qval_tip"] <= qval_thresh) &
                            (tip_base["expression_mean_base"] >= exp_thresh) &  (tip_base["expression_mean_tip"] >= exp_thresh),"gene"].tolist()

plot = tip_base.loc[tip_base["gene"].isin(signif_genes)]
 
genes = ["Ace2","Pigr","Cyb5r3","Net1","Apob","Lct","Cars2","Drg1","Hook2"]
genes = plot["gene"].tolist()

ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_base", "log2fc_tip",genes=genes,figsize=(6,6),
                                          repel=True,color_genes="black",title="Polarization across villi zones",
                         xlab="log2(nucleus/cytoplasm) - Base of villi",ylab="log2(nucleus/cytoplasm) - Tip of villi", 
                         x_line=0, y_line=0)

corr, pval = spearmanr(plot['log2fc_base'],  plot['log2fc_tip'])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95, f"r = {corr:.2f}, p = {pval:.2g}", transform=plt.gca().transAxes,color="black")

si.plot.save("tip_vs_base_retention_sctter",ax=ax)     
#%% Save + export
# si = HiVis.load(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\mouse_intestine.pkl")
si.save()


#%% Luminal apical vs nuclear apical
#%%% Import nuc map
si.add_annotations(r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - nuc_masks.geojson",
                   name="nuc_mask",measurements=False)

si.agg_from_annotations("nuc_mask_id",name="nuc_mask",geojson_path=r"X:\roy\viziumHD\analysis\Qupath\mouse_intestine\export\cropped - nuc_masks.geojson")

si.agg["nuc_mask"].plot.cells("nuc_mask")

#%%% Compute distance to nuc

# si.analysis.compute_distances("nuc_mask")
si_apical = si[si.adata.obs["apicome_classifier_region"].isin(["apical","nucleus"]),:]
si_apical.analysis.compute_distances("nuc_mask")
#%%% Assign "very apical" and "slightly apical"


dist = 4
si_apical.plot.spatial("dist_to_nuc_mask",xlim=[0,200],ylim=[3400,3500],alpha=0.5,show_zeros=1)
si_apical.plot.spatial("apicome_classifier_region",xlim=[0,200],ylim=[3400,3500],alpha=0.5,show_zeros=1)

si_apical.plot.hist("dist_to_nuc_mask",bins=50)

si_apical.adata.obs["apical_type"] = "nuc"
si_apical.adata.obs.loc[(si_apical["dist_to_nuc_mask"] > 0) & 
                        (si_apical["dist_to_nuc_mask"] < dist),"apical_type"] = "perinuc"

si_apical.adata.obs.loc[si_apical["dist_to_nuc_mask"] > dist,"apical_type"] = "luminal"

#%%% Plot spatial - distance to nucleus

xlim = [0,200]
ylim = [3430,3530]

fig, axes = plt.subplots(2,1,figsize=(7,7))

ax = si_apical.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Mouse intestine") 

ax = si_apical.plot.spatial("dist_to_nuc_mask",xlim=xlim,ylim=ylim,title=" ",
                     alpha=1,legend_title="",exact=True,ax=axes[1],
                     cmap="winter",legend="center left",scalebar={"text":False})
si_apical.agg["nuc_mask"].adata.obs["temp"] = "t"
ax = si_apical.agg["nuc_mask"].plot.cells("temp",cmap=["orange","orange"],legend=False,title=" ",image=False,
                                          alpha=0.3,line_color="orange",xlim=xlim,ylim=ylim,ax=axes[1])
HiVis.HiVis_plot.add_legend({"orange":"Nucleus"}, ax, title=None, loc="center left")

plt.tight_layout()

si_apical.plot.save("dist_to_nuc_SPATIAL",fig=fig)


fig, axes = plt.subplots(2,1,figsize=(7,7))

ax = si_apical.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Mouse intestine",legend=False) 

ax = si_apical.plot.spatial("dist_to_nuc_mask",xlim=xlim,ylim=ylim,title=" ",
                     alpha=1,legend_title="",exact=True,ax=axes[1],
                     cmap="winter",legend=False,scalebar={"text":False})
si_apical.agg["nuc_mask"].adata.obs["temp"] = "t"
ax = si_apical.agg["nuc_mask"].plot.cells("temp",cmap=["orange","orange"],legend=False,title=" ",image=False,
                                          alpha=0.3,line_color="orange",xlim=xlim,ylim=ylim,ax=axes[1])
HiVis.HiVis_plot.add_legend({"orange":"Nucleus"}, ax, title=None, loc="center left")

plt.tight_layout()

si_apical.plot.save("dist_to_nuc_SPATIAL_noLegend",fig=fig)

#%%% Plot spatial - classification

si_apical["temp"] = si_apical.adata.obs["apical_type"].str.capitalize()
si_apical.adata.obs.loc[(si_apical["temp"] == None) | (si_apical["temp"] == "Nuc"),"temp"] = np.nan
si_apical.update_meta("temp",{"Luminal":"Luminal apical","Nuc":"Nucleus","Perinuc":"Perinuclear apical"})
apicome_cmap = {"Luminal apical":"green","Perinuclear apical":"cyan","Nucleus":"orange"}

xlim = [0,200]
ylim = [3430,3530]

fig, axes = plt.subplots(2,1,figsize=(7,7))

ax = si_apical.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Mouse intestine") 

ax = si_apical.plot.spatial("temp",xlim=xlim,ylim=ylim,title=" ",
                     alpha=0.6,legend_title="",exact=True,ax=axes[1],
                     cmap=apicome_cmap,legend=False,scalebar={"text":False})
si_apical.agg["nuc_mask"].adata.obs["temp"] = "t"
ax = si_apical.agg["nuc_mask"].plot.cells("temp",cmap=["orange","orange"],legend=False,title=" ",image=False,
                                          alpha=0.3,line_color="orange",xlim=xlim,ylim=ylim,ax=ax)
HiVis.HiVis_plot.add_legend({v:k for k,v in apicome_cmap.items()}, ax, title=None, loc="center left")
plt.tight_layout()

si_apical.plot.save("apicome_extreme_SPATIAL",ax=ax)

#%%% DGE

si_apical_lumen = si_apical.analysis.dge("apical_type","perinuc","luminal")
# si_apical_nuc = si_apical.analysis.dge("apical_type","perinuc","nuc")

si_apical_lumen.to_csv(r"X:\roy\viziumHD\analysis\Matlab\ER\si_apical_lumen.csv")
# si_apical_nuc.to_csv(r"X:\roy\viziumHD\analysis\Matlab\ER\si_apical_nuc.csv")


# df1 = si_apical_lumen[['gene', 'log2fc', 'expression_mean','qval']].copy()
# df2 = si_apical_nuc[['gene', 'log2fc', 'expression_mean','qval']].copy()

# df = pd.merge(df1,df2,suffixes=["_cyto","_nuc"],on="gene")
# df = df.loc[:, ~df.columns.str.startswith("count")]
df = si_apical_lumen.copy()
df.to_csv(r"X:\roy\viziumHD\analysis\Matlab\ER\apical_types.csv")
df.rename(columns={c: c if c == "gene" else f"{c}_cyto" for c in df.columns}, inplace=True)

#%%% Plot scatter
qval_thresh = 1
exp_thresh = 3e-5

plot = df.copy()

# plot = df.loc[(df["qval_nuc"] <= qval_thresh) & (df["qval_cyto"] <= qval_thresh) ].copy()

plot["extreme_perinuc_ratio"] = -plot["log2fc_cyto"]


genes = plot.loc[plot["qval_cyto"] <= 0.05,"gene"].tolist()

plot["exp"] = np.log10(plot["expression_mean_cyto"])
plot = plot.loc[plot["expression_mean_cyto"] >= exp_thresh]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "exp", "extreme_perinuc_ratio",
                                          genes =genes ,repel=False,
                                          # genes=genes,genes2=genes2,repel=True,
                                     xlab="log10(expression)", ylab="log2(Luminal apical/Perinuclear apical)",y_line=0,
                                     title=" ")

#%% Quanify smFISH Atp1b1, Lct

import re
from pathlib import Path


MEASUREMENT_ORDER = [
    "microvil_Atp1b1",
    "microvil_Lct",
    "perinuc_Atp1b1",
    "perinuc_Lct",
    "back_Atp1b1",
    "back_Lct",
]


def read_table(path):
    seps = [",", "\t", ";"]
    for sep in seps:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")


def normalize(df):
    df = df.copy()
    first_col = df.columns[0]
    if str(first_col).startswith("Unnamed") or str(first_col).strip() == "":
        df = df.set_index(first_col)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def parse_mouse_fov(stem):
    m = re.match(r"^(r[^_ -]+)[_ -](\d+)$", stem.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename '{stem}' does not match pattern like r1_1")
    return m.group(1), m.group(2)


def build_results_table(folder, value_col="median", file_glob="*.csv"):
    folder = Path(folder)
    value_key = value_col.strip().lower()

    rows = []
    for path in sorted(folder.glob(file_glob)):
        if path.is_dir():
            continue

        mouse, fov = parse_mouse_fov(path.stem)

        df = normalize(read_table(path))
        if value_key not in df.columns:
            raise KeyError(f"{path.name}: column '{value_col}' not found. Columns: {list(df.columns)}")

        values = df[value_key].to_numpy()
        if len(values) < 6:
            raise ValueError(f"{path.name}: expected at least 6 rows, found {len(values)}")

        n_full = (len(values) // 6) * 6
        if n_full != len(values):
            values = values[:n_full]

        mtx = values.reshape(-1, 6)
        meas = pd.DataFrame(mtx, columns=MEASUREMENT_ORDER)
        meas.insert(0, "measurement", np.arange(1, len(meas) + 1, dtype=int))

        def score(num, den):
            out = np.full_like(num, np.nan, dtype=float)
            valid = (num > 0) & (den > 0)
            out[valid] = np.log2(num[valid] / den[valid])
            return out

        num_a = meas["microvil_Atp1b1"].to_numpy() - meas["back_Atp1b1"].to_numpy()
        den_a = meas["perinuc_Atp1b1"].to_numpy() - meas["back_Atp1b1"].to_numpy()
        s_a = score(num_a, den_a)

        num_l = meas["microvil_Lct"].to_numpy() - meas["back_Lct"].to_numpy()
        den_l = meas["perinuc_Lct"].to_numpy() - meas["back_Lct"].to_numpy()
        s_l = score(num_l, den_l)

        for i in range(len(meas)):
            rows.append({"measurement": int(meas.loc[i, "measurement"]), "score": s_a[i], "gene": "Atp1b1", "mouse": mouse, "FOV": fov})
            rows.append({"measurement": int(meas.loc[i, "measurement"]), "score": s_l[i], "gene": "Lct", "mouse": mouse, "FOV": fov})

    return pd.DataFrame(rows, columns=["measurement", "score", "gene", "mouse", "FOV"])

import seaborn as sns

folder = r"X:\roy\viziumHD\analysis\Python\smFISH_quant\Atp1b1_Lct"
res = build_results_table(folder, value_col="median", file_glob="*.csv")
# res.to_csv(Path(folder) / "results_scores.csv", index=False)
#%%%% Plot violin


from scipy.stats import wilcoxon
from matplotlib.collections import PolyCollection


res_plot = res.dropna(subset=["score"]).copy()
gene_order = ["Atp1b1", "Lct"]
res_plot = res_plot[res_plot["gene"].isin(gene_order)]

pairs = res_plot.pivot_table(index=["mouse", "FOV", "measurement"], columns="gene", values="score", aggfunc="first").dropna(subset=gene_order).reset_index()

rng = np.random.default_rng(0)
jitter = 0.03
xpos = {g:i for i, g in enumerate(gene_order)}
pairs["x_Atp1b1"] = xpos["Atp1b1"] + rng.uniform(-jitter, jitter, size=len(pairs))
pairs["x_Lct"] = xpos["Lct"] + rng.uniform(-jitter, jitter, size=len(pairs))

fig, ax = plt.subplots(1,1,figsize=(7,7))
sns.violinplot(data=res_plot, x="gene", y="score", order=gene_order, inner=None, ax=ax,fill=False,color="black",)
for _, r in pairs.iterrows():
    ax.plot([r["x_Atp1b1"], r["x_Lct"]], [r["Atp1b1"], r["Lct"]], linewidth=1, alpha=0.35, color="black")
    ax.scatter([r["x_Atp1b1"], r["x_Lct"]], [r["Atp1b1"], r["Lct"]], s=15, alpha=0.85, color="black")

meds = res_plot.groupby("gene")["score"].median().reindex(gene_order)
def xspan_of_poly_at_y(verts, y):
    xs = []
    for (x1, y1), (x2, y2) in zip(verts[:-1], verts[1:]):
        if y1 == y2:
            continue
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    if len(xs) < 2:
        return None, None
    return float(np.min(xs)), float(np.max(xs))


polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
centers = []
for c in polys:
    v = c.get_paths()[0].vertices
    centers.append(np.mean(v[:, 0]))

for i, g in enumerate(gene_order):
    y = float(meds[g])
    if len(polys) == 0:
        break
    j = int(np.argmin(np.abs(np.array(centers) - i)))
    verts = polys[j].get_paths()[0].vertices
    xmin, xmax = xspan_of_poly_at_y(verts, y)
    if xmin is not None:
        ax.hlines(y, xmin, xmax, colors="red", linewidth=2, zorder=6)
w = wilcoxon(pairs["Atp1b1"].to_numpy(), pairs["Lct"].to_numpy(), zero_method="wilcox", alternative="two-sided", mode="auto")
p = w.pvalue

ymax = np.nanmax(res_plot["score"].to_numpy())
ymin = np.nanmin(res_plot["score"].to_numpy())
yr = ymax - ymin if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
y_text = ymax + 0.06 * yr
ax.text(0.5, 0.9, f"p={p:.2g}", ha="center", va="bottom")

ax.set_ylim(ymin - 0.05 * yr, y_text + 0.05 * yr)
ax.set_ylabel("log2(luminal/perinuclear)")
ax.set_xlabel(None)
ax.set_title("Apical intensities in smFISH")
plt.tight_layout()

si_apical.plot.save("smFISH_quant",ax=ax)

#%% Compare manual annotation to pixel-classifier (rev. 2)


pixel = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\apicome_classifier_epi.csv",index_col=0)
manual = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\apicome_manual_epi.csv",index_col=0)
cols = ["log2fc","expression_mean","qval","gene"]
pixel, manual = pixel[cols], manual[cols]

#%%% Filter and merge
exp_thresh = 1e-5
qval_thresh = 0.05
comp = pd.merge(pixel, manual, on="gene",suffixes=["_pixel_classifier","_manual"])
comp = comp.loc[(comp["expression_mean_manual"]>exp_thresh) & 
                (comp["expression_mean_pixel_classifier"]>exp_thresh) &
                (comp["qval_manual"]<qval_thresh) & 
                (comp["qval_pixel_classifier"]<qval_thresh)]

#%%% Plot

ax = HiVis.HiVis_plot.plot_scatter_signif(comp, "log2fc_pixel_classifier","log2fc_manual",figsize=(6,6),
                                     x_line=0,y_line=0, title=" ",
                                     xlab="Log2(apical/basal) - pixel classifier",ylab="Log2(apical/basal) - manual annotation")

corr, pval = spearmanr(comp['log2fc_pixel_classifier'],  comp['log2fc_manual'])
pval = pval if pval > 0 else 1e-300
ax.text(0.05, 0.95, f"r = {corr:.2f}, p = {pval:.2g}", transform=plt.gca().transAxes,color="black")
# si.plot.save("comparison_manual_pixel_classifier_apicome", ax=ax)
plt.savefig(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\mouse_intestine_comparison_manual_pixel_classifier_apicome.pdf",dpi=300)
plt.savefig(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_intestine\output\mouse_intestine_comparison_manual_pixel_classifier_apicome.png",dpi=300)
