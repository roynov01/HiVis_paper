# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:48:01 2025

@author: royno
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from jenkspy import jenks_breaks
import matplotlib.pyplot as plt
# import harmonypy as hm
import matplotlib.cm as cm
import seaborn as sns
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
HiVis.HiVis_plot.DEFAULT_SAVE_FORMATS = ["png","pdf","svg"]


def ofras_pipeline_liver(input_df, vizium_instance, additional_obs=None):
    '''
    Adds the info from Ofras pipeline to the vizium.adata.obs. 
    Return dict of cell_ID:[nucs].
    parameters:
        * input_df - DataFrame of Ofras pipeline
        * additional_obs - which columns besides "Cell_ID" from Ofras CSV to add to the vizium.adata.obs
    '''
    print("[Processing dataframe]")
    for col in ["Object type","Object ID","Name","InNuc","InCell","Classification"]:
        if col not in input_df.columns:
            raise ValueError('columns of dataframe must include: ["Object type","Object ID","Name","InNuc","InCell","Classification"]')
            
    df = input_df.copy()
    if additional_obs is None:
        additional_obs = []
    
    # Split only the spots
    spots_only = df[df["Object type"] == "Tile"] # spots
    split_names = spots_only['Name'].str.split('__|\*\*', n=1, expand=True)
    split_names.columns = ['Spot_ID', 'Cell_ID']
    split_names[['Cell_ID', 'Nuc_ID']] = split_names['Cell_ID'].str.split('\+\+', n=1, expand=True)
    
    # Delete assignment of problematic spots (that are assigned to two nuclei or cells)
    #split_names.loc[split_names["Spot_ID"].str.len() > 50,["Cell_ID","Nuc_ID"]] = np.nan
    
    # Assign spots with more than one cell to the first one.
    split_names['Cell_ID'] = split_names['Cell_ID'].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else x
    )
    split_names['Nuc_ID'] = split_names['Nuc_ID'].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) and '_' in x else x
    )
    
    spots_only = spots_only.join(split_names)
    spots_only.loc[spots_only["Classification"] == "Spot", "Classification"] = np.nan

    # Which columns to add from spots, and which from cells:
    cols = ["Cell_ID", "Spot_ID","Nuc_ID","InCell", "InNuc","Classification"] + additional_obs 
    cols = list(set(cols))
    for col in cols:
        if col in vizium_instance.adata.obs.columns:
            print(f"[Column '{col}' already exists. Overwriting...]")
            vizium_instance.adata.obs.drop(columns=[col],inplace=True)
    nans = spots_only[cols].isna().all()
    cols_to_add_from_cells = nans.index[nans == True].tolist() 
    cols_to_add_from_spots = [col for col in spots_only[cols].columns if col not in cols_to_add_from_cells]
    
    # Add spot-metadata
    spots_only = spots_only[cols_to_add_from_spots]
    spots_only = spots_only.set_index("Spot_ID")
    
    vizium_instance.adata.obs = vizium_instance.adata.obs.join(spots_only,how='left')
    
    # Add cell-metadata
    cells_only = df.loc[df["Object type"] == "Cell",cols_to_add_from_cells + ['Object ID']]
    cells_only.rename(columns={"Object ID": "Cell_ID"},inplace=True)
    cells_only = cells_only.set_index("Cell_ID")
    vizium_instance.adata.obs = vizium_instance.adata.obs.join(cells_only,how="left", on="Cell_ID")
    

    # Find which nucs are in each cell, return dict of "cell_ID":[ind_nuc1]
    nucs_only = df[df["Object type"] == "Detection"]
    split_names = nucs_only['Name'].str.split('__', n=1, expand=True)
    nucs_only['Cell_ID'] = split_names[1]
    grouped_nucs = (nucs_only.groupby("Cell_ID")["Object ID"].apply(lambda names: 
                           pd.Series(names.iloc[:2].values)).unstack())
    grouped_nucs.columns = ["nuc1", "nuc2"]
    cells_nuc = cells_only.merge(grouped_nucs, on="Cell_ID", how="left")[["nuc1","nuc2"]]

    return cells_nuc


def _aggregate_data_two_nuclei(adata, cells_nuc, group_col=None,
    group_col_cell="Cell_ID",group_col_nuc="Nuc_ID",in_cell_col="InCell",
    nuc_col="InNuc"):
    '''
    Helper function that can be used for as "aggregation_func" in new_adata().
    Aggregates expression data based on processed dataframe from 
    Ofras pipeline that uses Cellpose for liver (cells with 0,1,2 nuclei).     
    '''
    from tqdm import tqdm
    from scipy.sparse import lil_matrix
    # Filter only spots that are inside a cell
    mask = adata.obs[in_cell_col] == 1
    adata_filtered = adata[list(mask[mask].index)].copy()
    
    # Subset: nucleus spots vs cytoplasm spots
    adata_nuc = adata_filtered[adata_filtered.obs[nuc_col] == 1].copy()
    adata_cyto = adata_filtered[adata_filtered.obs[nuc_col] == 0].copy()
    
    # Build dictionary of indices, but now grouped by Nuc_ID for nuclear subset
    nuc_dict = adata_nuc.obs.groupby(group_col_nuc).indices
    cyto_dict = adata_cyto.obs.groupby(group_col_cell).indices
    
    # Determine cell IDs we want to process
    cells_ids = cells_nuc.index.tolist()
    num_cells = len(cells_ids)
    num_genes = adata_filtered.shape[1]
    
    # Allocate sparse matrices
    nucleus_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    cyto_data = lil_matrix((num_cells, num_genes), dtype=np.float32)
    
    # Preallocate double nucleated nucs and cells for analysis
    double_nuc_cells = cells_nuc.index[~cells_nuc["nuc2"].isna() & 
                                       cells_nuc["nuc2"].isin(nuc_dict.keys())]
    num_nucs_in_2nuc_cells = len(double_nuc_cells) * 2
    nucs_2nuc_cells = lil_matrix((num_nucs_in_2nuc_cells, num_genes), dtype=np.float32)
    cells_2nucs_list = [None for _ in range(num_nucs_in_2nuc_cells)]
    nucs_2nucs_list = [None for _ in range(num_nucs_in_2nuc_cells)]
    nuc_index = 0
    
    for i, cell in enumerate(tqdm(cells_ids, desc='Aggregating spots expression')):
        # Aggregate cytoplasm for this cell 
        if cell in cyto_dict:
            cyto_data[i, :] = adata_cyto[cyto_dict[cell], :].X.sum(axis=0)
        
        nuc1_id = cells_nuc.loc[cell, "nuc1"]
        nuc2_id = cells_nuc.loc[cell, "nuc2"]
        
        # Single nucleus
        if pd.notna(nuc1_id) and (nuc1_id in nuc_dict):
            nuc1_expr = adata_nuc[nuc_dict[nuc1_id], :].X.sum(axis=0)
        
        # No nuclei
        else: 
            nuc1_expr = 0
        
        # Two nuclei
        if pd.notna(nuc2_id) and (nuc2_id in nuc_dict):
            nuc2_expr = adata_nuc[nuc_dict[nuc2_id], :].X.sum(axis=0)
            
            # Add both nuclei for 2nuc analysis
            nucs_2nucs_list[nuc_index] = nuc1_id
            cells_2nucs_list[nuc_index] = cell
            nucs_2nuc_cells[nuc_index,:] = nuc1_expr
            nuc_index += 1
            nucs_2nucs_list[nuc_index] = nuc2_id
            cells_2nucs_list[nuc_index] = cell
            nucs_2nuc_cells[nuc_index,:] = nuc2_expr
            nuc_index += 1
        else:
            nuc2_expr = 0
        # Total nucleus expression is expression in both nuclei
        nucleus_data[i, :] = nuc1_expr + nuc2_expr
    
    # Calculate which genes are enriched in one nuclei out of both
    genes = np.array(adata.var_names)
    df = pd.DataFrame(nucs_2nuc_cells.tocsr().toarray(),index=nucs_2nucs_list,columns=genes)
    df["Cell_ID"] = cells_2nucs_list
    nuc_cell_dict = df.groupby("Cell_ID").indices
    del df["Cell_ID"]
    
    
    cyto_data = cyto_data.tocsr()
    nucleus_data = nucleus_data.tocsr()
    cell_data = nucleus_data + cyto_data
    layers = {"nuc":nucleus_data, "cyto": cyto_data}
    
    return cell_data, cells_ids, layers, {"nuc_by_genes":df,"nuc_cell_dict":nuc_cell_dict}

def plot_zonation(agg,gene,column="zone",layer=None, 
                  color="blue",figsize=(7,7), ax=None):
    adata = agg.adata
    if layer is not None:
        expr_data = adata[:, gene].layers[layer]
    else:
        expr_data = adata[:, gene].X
    
    if hasattr(expr_data, "toarray"):
        expr_data = expr_data.toarray()
    expr_data = np.ravel(expr_data)

    df = pd.DataFrame({"expression": expr_data,"zone": adata.obs[column].values})
    grouped = df.groupby("zone")["expression"]
    means = grouped.mean()
    n = grouped.count()
    se = grouped.std() / np.sqrt(n)
    zones_sorted = sorted(means.index)
    means_sorted = means.loc[zones_sorted]
    se_sorted = se.loc[zones_sorted]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(zones_sorted, means_sorted, color=color, label='Mean expression')
    ax.fill_between(zones_sorted,means_sorted - se_sorted,means_sorted + se_sorted,
        color=color,alpha=0.2,label='±1 SE')
    ax.set_xlabel(column)
    ax.set_ylabel("Exression")
    ax.set_title(f"{gene}")
    return ax

#%% Import data
wt98 = HiVis.load(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT.pkl")
wt97 = HiVis.load(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_97_WT.pkl")


wt98.recolor({'ATP1A1': 'red', 'CD31': None, 'autofluorescence': None, 'DAPI': 'blue'})
wt98.plot.spatial(save=True,img_resolution="full")
wt97.plot.spatial(save=True,img_resolution="full")
#%% Aggregate

additional_obs=["Detection probability","Nucleus: Circularity",
                "dist_to_bv_um",
                "Nucleus: Solidity","Cell: Area µm^2","Cell: Circularity",
                "Cell: Solidity","DistToCell","DistToNuc",
                "Distance to annotation empty µm","nNucSpots","Nucleus: Area µm^2",
                "Nucleus/Cell area ratio","blood_vessels_fullres: Blood_vessel %",
                "blood_vessels_fullres: hepato %","blood_vessels_fullres: empty %"]
cols = ["autofluorescence", "DAPI","CD31","Classification"]

markers_PC = ['Akr1c6','Alad','Blvrb','C6','Car3','Ccdc107','Cml2','Cyp2c68','Cyp2d9','Cyp3a11','Entpd5','Fmo1','Gsta3','Gstm1','Gstm6','Gstt1','Hpd','Hsd17b10','Inmt','Iqgap2','Mgst1','Nrn1','Pex11a','Pon1','Psmd4','Slc22a1','Tex264']
markers_PP = ['Afm','Aldh1l1','Asl','Ass1','Atp5a1','Atp5g1','C8a','C8b','Ces3b','Cyp2f2','Elovl2','Fads1','Fbp1','Ftcd','Gm2a','Hpx','Hsd17b13','Ifitm3','Igf1','Igfals','Khk','Mug2','Pygl','Sepp1','Serpina1c','Serpina1e','Serpind1','Vtn']

genes_to_plot = ['Nt5e','Glul','Oat','Lgr5','Axin2','Wnt2','Rspo3',
       'Cdh2','Cyp1a2','Cyp2e1','Cyp7a1','Hsp90aa1','Dcn','Npr2',
       'Igfbp2','Hamp','Hamp2',"Hnf4a","Srebf1","Hgf","Notch3",
       'Epcam','Spp1','Sds','Alb','Cyp2f2','Cdh1','Sox9',
       'G6pc','Pck1','Fasn','Acly','Ass1','Asl','Gls2']

for viz, sample in zip([wt98, wt97], ["98", "97"]):
    path = rf"A:\royno\Visium_HD_liver\experiment1\qupath_project\results_v2\mouse_liver_{sample}_WT_fullres.tif - 2d copy for alternative analysis_detections.csv"
    path = rf"A:\royno\HiVis_proj_v2\Qupath6\results_mouse_liver\mouse_liver_{sample}_WT_fullres.tif - Cellpose_CustomDoubleNucleatedScript_Ch14_detections.csv"

    input_df_liver = pd.read_csv(path,sep="\t")
    
    input_df_liver.rename(columns={"Distance to annotation Blood_vessel µm":"dist_to_bv_um"},inplace=True)

    # Aggregate
    cells_nuc = ofras_pipeline_liver(input_df=input_df_liver, vizium_instance=viz,additional_obs=additional_obs)
    
    adata_sc, preprocess_results = HiVis.Aggregation_utils.new_adata(viz.adata, "Cell_ID", _aggregate_data_two_nuclei,
                                           obs2agg=cols,cells_nuc=cells_nuc)
    
    input_df_liver_indexed = input_df_liver.loc[input_df_liver["Object type"]=="Cell"]
    input_df_liver_indexed.rename(columns={"Object ID":"Cell_ID"},inplace=True)
    input_df_liver_indexed.set_index("Cell_ID",inplace=True)
    adata_sc.obs = adata_sc.obs.join(input_df_liver_indexed[additional_obs], how="left")

    viz.preprocess_results = preprocess_results
    viz.cells_nuc = cells_nuc
    
    viz.add_agg(adata_sc, "SC")
    
    # Map nucs to cells
    map_df = input_df_liver.loc[input_df_liver["Classification"] == "NucInCell",["Object ID","Name"]]
    map_df["Cell"] = [row.split("null__")[1] for row in map_df["Name"]]
    adata = viz.agg["SC"].adata
    map_df = map_df[map_df["Cell"].isin(adata.obs_names)].copy()
    adata_nuc = adata[map_df["Cell"].values].copy()
    adata_nuc.obs.index = map_df["Object ID"].values
    adata_nuc.obs.index.name = "Cell_ID"
    viz.add_agg(adata_nuc, "nuc")
    path=rf"A:\royno\Visium_HD_liver\experiment1\qupath_project\results_v2\mouse_liver_{sample}_WT_fullres.tif - 2d copy for alternative analysis_nucInHepatoCells.geojson"
    path=rf"A:\royno\HiVis_proj_v2\Qupath6\results_mouse_liver\mouse_liver_{sample}_WT_fullres.tif - Cellpose_CustomDoubleNucleatedScript_Ch14_nucInHepatoCells.geojson"
    viz.agg["nuc"].import_geometry(path,object_type="detection")
    
    # Add cells geometry
    print(f"Adding geometry: {sample}")
    path = rf"A:\royno\Visium_HD_liver\experiment1\qupath_project\results_v2\mouse_liver_{sample}_WT_fullres.tif - 2d copy for alternative analysis_cells.geojson"
    path = rf"A:\royno\HiVis_proj_v2\Qupath6\results_mouse_liver\mouse_liver_{sample}_WT_fullres.tif - Cellpose_CustomDoubleNucleatedScript_Ch14_cells.geojson"
    
    viz.agg["SC"].import_geometry(path)
    viz.agg["HEP"] = viz.agg["SC"][viz.agg["SC"].adata.obs["Classification"] != "BV-Cell",:]
    
    # Calculate Eta score
    markers_PC = [g for g in markers_PC if g in viz.agg["SC"].adata.var_names]
    markers_PP = [g for g in markers_PP if g in viz.agg["SC"].adata.var_names]
    sum_pp = viz.agg["HEP"].adata[:,markers_PP].X.sum(axis=1)
    sum_pc = viz.agg["HEP"].adata[:,markers_PC].X.sum(axis=1)
    viz.agg["HEP"].adata.obs["eta"] = sum_pp / (sum_pp + sum_pc)
    viz.agg["HEP"].plot.spatial("eta",cmap="viridis",image=False,save=True,size=5)
    viz.agg["HEP"].plot.hist("eta",save=True,xlab="Eta score",bins=20)
    plt.show()
    viz.agg["HEP"] = viz.agg["HEP"][~viz.agg["HEP"].adata.obs["eta"].isna(),:]
    
    # Bin zones
    print(f"Binning zones: {sample}")
    num_bins = 6
    eta_values = viz.agg["HEP"].adata.obs["eta"].values
    breaks = jenks_breaks(eta_values, n_classes=num_bins)
    zone_labels = range(1, num_bins + 1)
    viz.agg["HEP"].adata.obs["zone_non_smooth"] = pd.cut(eta_values, bins=breaks, labels=zone_labels,include_lowest=True)
    
    fig, axes = plt.subplots(1,3,figsize=(21,7))
    viz.agg["HEP"].plot.hist("zone_non_smooth",ax=axes[0],title="Fisher-jenks binning")
    viz.agg["HEP"].plot.spatial("zone_non_smooth",ax=axes[1],axis_labels=False,
        cmap="viridis",image=False, legend=False,size=5,title="Fisher-jenks binning")
    viz.agg["HEP"].plot.hist("eta", ax=axes[2], xlab="Eta score",title="Fisher-jenks binning")
    for b in breaks:
        axes[2].axvline(x=b, color='red', linestyle='--')
    plt.tight_layout()
    viz.agg["HEP"].plot.save("zonation_binning",fig=fig)

    # Smooth the zones
    print(f"Smoothing zones: {sample}")
    radius = 40
    method = "median"
    viz.agg["HEP"].adata.obs["zone_non_smooth"] = viz.agg["HEP"].adata.obs["zone_non_smooth"].astype(np.int8)
    _ = viz.agg["HEP"].analysis.smooth("zone_non_smooth",radius=radius,method=method,new_col_name="zone")
    viz.agg["HEP"]["zone"] = round(viz.agg["HEP"].adata.obs["zone"])
    viz.agg["HEP"].sync("zone")
    zone_series = viz.agg["HEP"].adata.obs["zone"]
    viz.agg["SC"].adata.obs["zone_hep"] = zone_series.reindex(viz.agg["SC"].adata.obs.index)
    _ = viz.agg["SC"].analysis.smooth("zone_hep",radius=radius,method=method,new_col_name="zone")
    viz.agg["SC"]["zone"] = round(viz.agg["SC"].adata.obs["zone"])
    
    # Plot zones
    ax = viz.agg["HEP"].plot.spatial("zone",size=5,image=False,scalebar={"color":"black"},
                                     title="",cmap="viridis",legend_title='',rasterize=False)
    if sample=="98":
        ax.text(1.04,0.825,"Portal",transform=plt.gca().transAxes,color="black")
        ax.text(1.04,0.16,"Central",transform=plt.gca().transAxes,color="black")
    else:
        ax.text(1.03,0.925,"Portal",transform=plt.gca().transAxes,color="black")
        ax.text(1.03,0.04,"Central",transform=plt.gca().transAxes,color="black")   
        
    viz.agg["HEP"].plot.save("zone",ax=ax)
    
    # Matnorm
    viz.agg["HEP"].adata.layers["matnorm"] = HiVis.HiVis_utils.matnorm(viz.agg["HEP"].adata.X,axis="row")
    viz.agg["SC"].adata.layers["matnorm"] = HiVis.HiVis_utils.matnorm(viz.agg["SC"].adata.X,axis="row")
    
    # Plot zonation genes
    fig, axes = plt.subplots(nrows=5, ncols=7, figsize=(21, 15))
    for gene, ax in zip(genes_to_plot, axes.flat):
        cur_ax = plot_zonation(viz.agg["HEP"],gene=gene,layer="matnorm",ax=ax)
        cur_ax.set_title(gene)
        cur_ax.set_xlabel(None)
        cur_ax.set_ylabel(None)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    viz.agg["HEP"].plot.save("zonation_examples",fig=fig)

    print(f"Saving: {sample}")
    # viz.save()

#%%  plot cells
xlim = [1050,1550]
ylim = [2100,2350]
t = wt98.crop(xlim=xlim,ylim=ylim)
t.agg["SC"]["temp"] = t.agg["SC"]["Classification"] == "BV-Cell"

fig, axes = plt.subplots(2,1,figsize=(7,7))

ax = t.plot.spatial(scalebar={"text_offset":0.06},ax=axes[0],title="Single cell segmentation")
ax = t.agg["SC"].plot.cells("temp",line_color="green",legend=False,scalebar=False,cmap=["green","green"],ax=axes[1])
ax = t.agg["HEP"].plot.cells(line_color="white",ax=ax,scalebar={"text":False})
ax = t.agg["nuc"].plot.cells(line_color="orange",legend=False,scalebar=False,
                                xlim=xlim,ylim=ylim,ax=axes[1])
ax.set_title(None)
HiVis.HiVis_plot.add_legend({"red":"ATP1A1","blue":"DAPI","green":"NPC"},ax)
plt.tight_layout()
# wt98.plot.save("cells_segmentation",ax=ax)



xlim = [1050,1550]
ylim = [2100,2350]

fig, axes = plt.subplots(2,1,figsize=(7,7))
ax = t.plot.spatial(scalebar={"text_offset":0.06},ax=axes[0],title="Mouse liver")
t.agg["HEP"].adata.obs["temp"] = True
ax = t.agg["HEP"].plot.cells("temp",line_color="black",scalebar={"text":False},alpha=0.4,
                             cmap=["orange","orange"],legend=False,ax=axes[1])
wt98.agg["nuc"].adata.obs["temp"] = True
ax = wt98.agg["nuc"].plot.cells("temp",line_color="black",legend=False,scalebar=False,alpha=0.4,
                                xlim=xlim,ylim=ylim,ax=axes[1],image=False,cmap=["cyan","cyan"])
HiVis.HiVis_plot.add_legend({"orange":"Cytoplasm","cyan":"Nucleus"},ax)

ax.set_title(None)
ax.set_title(None)
plt.tight_layout()
wt98.plot.save("nucs_segmentation",ax=ax)


#%%% Plot cells and nucs
xlim = [1050,1550]
ylim = [2100,2350]

fig, axes = plt.subplots(2,1,figsize=(7,7))
ax = wt98.plot.spatial(xlim=xlim,ylim=ylim,axis_labels=False,ax=axes[0], title="Single cell segmentation",scalebar={"text_offset":0.08})
# ax.set_title(None)
ax = wt98.plot.spatial(xlim=xlim,ylim=ylim,axis_labels=False,scalebar=False,ax=axes[1])
wt98.agg["SC"]["temp"] = wt98.agg["SC"]["Classification"] == "BV-Cell"
ax = wt98.agg["SC"].plot.cells("temp",xlim=xlim,ylim=ylim,line_color="green",ax=ax,scalebar=False,
                               image=True,cmap=["green","green"],legend=False)
ax = wt98.agg["HEP"].plot.cells(xlim=xlim,ylim=ylim,line_color="white",ax=ax,scalebar=False,image=False)
ax = wt98.agg["nuc"].plot.cells(xlim=xlim,ylim=ylim,line_color="black",image=True,scalebar={"text":False},ax=ax)
HiVis.HiVis_plot.add_legend({"red":"ATP1A1","blue":"DAPI","green":"NPC"},ax)

ax.set_title(None)
plt.tight_layout()
wt98.plot.save("cells_and_nuclei_CELLS_poster",ax=ax)
wt98.plot.save("cells_segmentation",ax=ax)

#%% Plot zonation examples
gene1, gene2 = "Glul","Sds"
color1, color2 = "red","blue"
viz = wt98
exp_thresh = 2e-4

# fig, axes = plt.subplots(1,2,figsize=(14,7))

ax = plot_zonation(viz.agg["HEP"],gene=gene1,layer="matnorm",color=color1,figsize=(5,5))
ax = plot_zonation(viz.agg["HEP"],gene=gene2,layer="matnorm",ax=ax,color=color2,figsize=(5,5))
ax.axhline(y=exp_thresh,color="k", linestyle="--")
ax.text(0.1, 0.85,gene1,transform=ax.transAxes,color=color1)
ax.text(0.8, 0.35,gene2,transform=ax.transAxes,color=color2)
ax.set_xlabel("Zone")
ax.set_ylabel("Normilized expression")
ax.set_title("Example for zonated genes")
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xticklabels(["Central", "", "", "", "", "Portal"])
ax.set_xlabel(" ")
wt98.plot.save("zonation_genes_example1",ax=ax)

viz.agg["HEP"]["temp"] = viz.agg["HEP"].get(gene1,layer="matnorm") > exp_thresh
ax = viz.agg["HEP"].plot.spatial("temp",img_resolution="high",size=5,image=False,alpha=0.5,
                                legend=False,cmap=[color1,color1],scalebar=False)
viz.agg["HEP"]["temp"] = viz.agg["HEP"].get(gene2,layer="matnorm") > exp_thresh
ax = viz.agg["HEP"].plot.spatial("temp",img_resolution="high",size=5,ax=ax,legend=False,alpha=0.5,
                                 image=False,cmap=[color2,color2],title="",scalebar={"text_offset":0.04})
import matplotlib.patches as mpatches
patch = mpatches.Patch(color=color1, label=gene1)
patch2 = mpatches.Patch(color=color2, label=gene2)
ax.legend(handles=[patch,patch2], title=None, loc='lower right',bbox_to_anchor=(0.75,0) )

# plt.tight_layout()
# wt98.plot.save("zonation_genes_example",fig=fig)
wt98.plot.save("zonation_genes_example2",ax=ax)


#%% Merge both batches
both = wt98.agg["HEP"] + wt97.agg["HEP"]
both.rename("WT",full=True)
both.path_output = "X:\\roy\\viziumHD\\analysis\\python\\version_11\\organs\\mouse_liver\\output/WT"


both_sc = wt98.agg["SC"] + wt97.agg["SC"]
both_sc.rename("WT",full=True)
both_sc.path_output = "X:\\roy\\viziumHD\\analysis\\python\\version_11\\organs\\mouse_liver\\output/WT"
#%%% Plot QC
both_sc["nUMI_log10"] = np.log10(both_sc["nUMI"])

both_sc["temp"] = both_sc["Classification"]
both_sc.update_meta("temp",{"BV-Cell":"NPC","Cell-twoNuc":"Hepatocyte",
                            "Cell-oneNuc":"Hepatocyte","Cell-noNuc":"Hepatocyte"})
both_sc["temp2"] = both_sc["source_"]
both_sc.update_meta("temp2",{"mouse_liver_98_WT_SC":"Mouse 1","mouse_liver_97_WT_SC":"Mouse 2"})
df = both_sc.adata.obs[["nUMI_log10", "spot_count", "temp", "temp2"]]

fig, axes = plt.subplots(1, 2, figsize=(6, 6))

# sns.violinplot(data=df, x="temp", y="nUMI_log10", hue="temp2", split=True, ax=axes[0], inner="box",legend=False,
#                cut=0, scale="width", linewidth=0.8, palette="Set1", width=1)
sns.violinplot(
    data=df, x="temp", y="nUMI_log10", hue="temp2", split=True,legend=False,
    ax=axes[0], inner=None, cut=0, scale="width", linewidth=0.8, palette=["lightgreen","lightblue"]
)

sns.boxplot(
    data=df, x="temp", y="nUMI_log10", hue="temp2", dodge=True,showfliers=False,legend=False,
    ax=axes[0], width=0.3, showcaps=False, boxprops={"facecolor":"none", "zorder":3},
    whiskerprops={"linewidth":0}, medianprops={"color":"black", "linewidth":1.2}
)
axes[0].set_xlabel(None)
axes[0].set_ylabel("log10(nUMI per cell)")

sns.violinplot(
    data=df, x="temp", y="spot_count", hue="temp2", split=True,
    ax=axes[1], inner=None, cut=0, scale="width", linewidth=0.8, palette=["lightgreen","lightblue"]
)

sns.boxplot(
    data=df, x="temp", y="spot_count", hue="temp2", dodge=True,showfliers=False,legend=False,
    ax=axes[1], width=0.3, showcaps=False, boxprops={"facecolor":"none", "zorder":3},
    whiskerprops={"linewidth":0}, medianprops={"color":"black", "linewidth":1.2}
)
axes[1].set_xlabel(None)
axes[1].set_ylabel("Bins per cell")
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].legend(title=None)

plt.tight_layout()
both_sc.plot.save("QC_violins",fig=fig)
#%%% Check correlation between both mice
#%%%% Expression
df = both.analysis.pseudobulk("source_",layer="matnorm")

batches = ["Mouse 1", "Mouse 2"]
df.columns = batches

pn = df[df > 0].min().min()

df = np.log10(df+pn)
ax = HiVis.HiVis_plot.plot_scatter_signif(df, batches[0], batches[1],
                                  xlab=f"log10(expression) - {batches[0]}",title="Correlation between two samples",
                                  ylab=f"log10(expression) - {batches[1]}",size=20,
                                  color="black",figsize=(6,6))
corr, pval = spearmanr(df[batches[0]], df[batches[1]])
pval = pval if pval > 0 else 1e-300
ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}", transform=ax.transAxes)

both.plot.save("batch_correlation",ax=ax)

#%%%% Zones
def compute_com(df):
    columns = np.array(df.columns, dtype=float)  # Convert column names to numeric
    com = (df * columns).sum(axis=1) / df.sum(axis=1)
    return com

exp_thresh = 5e-5
exp_thresh = 1e-4
selected_genes = ["Glul","Oat","Npr2","Cyp1a2","Cyp2e1","Cyp7a1","Hamp","Hamp2","Hnf4a","Asl","Cyp2f2","Sox9","Alb","Sds"]

pb98 = wt98.agg["HEP"].analysis.pseudobulk("zone")


wt98.agg["HEP"].analysis.pseudobulk(layer="matnorm")

pb98 = pb98.loc[wt98.agg["HEP"].analysis.pseudobulk(layer="matnorm") > exp_thresh]

pb97 = wt97.agg["HEP"].analysis.pseudobulk("zone")
pb97 = pb97.loc[wt97.agg["HEP"].analysis.pseudobulk(layer="matnorm") > exp_thresh]

common_genes = pb97.index.intersection(pb98.index)
pb98 = pb98.loc[common_genes]
pb97 = pb97.loc[common_genes]

com1 = compute_com(pb97)
com2 = compute_com(pb98)

df = pd.DataFrame({batches[0]:com1,batches[1]:com2,"gene":com2.index},index=com2.index)

ax = HiVis.HiVis_plot.plot_scatter_signif(df, batches[0], batches[1],
                                  xlab=f"{batches[0]}",
                                  ylab=f"{batches[1]}",figsize=(6,6),
                                  genes=selected_genes,size=20,
                                  color="black")

corr, pval = spearmanr(df[batches[0]], df[batches[1]])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}", transform=ax.transAxes)

ax.set_xticks([df[batches[0]].min(), df[batches[0]].max()])
ax.set_xticklabels(["Central", "Portal"])
ax.set_yticks([df[batches[1]].min(), df[batches[1]].max()])
ax.set_yticklabels(["Central", "Portal"], rotation=90,va="center")

both.plot.save("COM correlation batches",ax=ax)

#%% Correlation with other datasets
#%%% correlation with Bahar K. et al. nature, 2017
exp_thresh = 5e-5
exp_thresh = 1e-4

bahar = pd.read_csv(r"X:\roy\viziumHD\analysis\python\version_11\organs\mouse_liver\input\bahar_2017.csv",index_col=0)
bahar.columns = [col.replace("Layer.","") for col in bahar.columns]

zonation = both.analysis.pseudobulk("zone")
zonation = zonation.loc[zonation.sum(axis=1) > exp_thresh]

common_genes = bahar.index.intersection(zonation.index)
bahar = bahar.loc[common_genes]
zonation = zonation.loc[common_genes]

high_exp_genes = bahar.index
high_exp_genes = bahar.index[bahar.max(axis=1)>exp_thresh]

com_bahar = compute_com(bahar.loc[high_exp_genes])
com_zonation = compute_com(zonation.loc[high_exp_genes])


df = pd.DataFrame({"VisiumHD":com_zonation,"SingleCell":com_bahar,"gene":high_exp_genes},
                  index=high_exp_genes)

ax = HiVis.HiVis_plot.plot_scatter_signif(df, "SingleCell", "VisiumHD",
                                  xlab="scRNA-seq",figsize=(6,6),
                                  ylab="VisiumHD",repel=True,color_genes="red",
                                  genes=selected_genes,size=20,
                                  color="black")

corr, pval = spearmanr(df["SingleCell"], df["VisiumHD"])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}", 
        transform=ax.transAxes)
ax.set_title("Correlation with Bahar Halpern et al. (2017)")
positions = ax.get_xticks()  
labels = [""] * len(positions)
labels[1] = "Central"
labels[-3] = "Portal"
ax.set_xticklabels(labels) 
positions_y = ax.get_yticks()
labels_y = [""] * len(positions_y)
labels_y[2] = "Central"
labels_y[-3] = "Portal"
ax.set_yticklabels(labels_y, rotation=90)

both.plot.save("COM correlation bahar",ax=ax)

#%%% Jiangshan Xu et al, Nature genetics, 2024
selected_genes = ["Glul","Oat","Npr2","Cyp1a2","Cyp2e1","Cyp7a1","Hamp","Hamp2","Hnf4a","Asl","Cyp2f2","Sox9","Alb","Sds"]

xu = pd.read_csv(r"X:\roy\viziumHD\analysis\python\version_11\organs\mouse_liver\input\Jiangshan_2024.csv",index_col=0)
xu.columns = [col.replace("Layer ","") for col in xu.columns]
xu = xu.loc[:, ~xu.columns.str.contains(r"^Unnamed|^pval|^qval", regex=True)]

zonation = both.analysis.pseudobulk("zone")
zonation = zonation.loc[zonation.sum(axis=1) > exp_thresh]

common_genes = xu.index.intersection(zonation.index)
xu = xu.loc[common_genes]
zonation = zonation.loc[common_genes]

com_xu = compute_com(xu)
com_zonation = compute_com(zonation)

df = pd.DataFrame({"VisiumHD":com_zonation,"StereoSeq":com_xu,"gene":common_genes},
                  index=common_genes)

ax = HiVis.HiVis_plot.plot_scatter_signif(df, "StereoSeq", "VisiumHD",
                                  xlab="Stereo-seq",figsize=(6,6),
                                  ylab="VisiumHD",repel=True,color_genes="red",
                                  genes=selected_genes,size=20,
                                  color="black")

corr, pval = spearmanr(df["StereoSeq"], df["VisiumHD"])
pval = pval if pval > 0 else 1e-300
ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}", transform=ax.transAxes)
ax.set_title("Correlation with Xu et al. (2024)")

positions = ax.get_xticks()  
labels = [""] * len(positions)
labels[1] = "Central"
labels[-3] = "Portal"
ax.set_xticklabels(labels) 
positions_y = ax.get_yticks()
labels_y = [""] * len(positions_y)
labels_y[2] = "Central"
labels_y[-2] = "Portal"
ax.set_yticklabels(labels_y, rotation=90)

both.plot.save("COM correlation stereoseq",ax=ax)

to_save = zonation.copy()
to_save = to_save.reindex(sorted(to_save.columns), axis=1)
to_save.columns = [f"Zone {int(z)}" for z in to_save.columns]
to_save["COM"] = com_zonation
both.plot.save("COM",fig=to_save)


#%% Ploidity probability across zones

from scipy.signal import find_peaks

viz = both
bins = 50
kernel_smoothing_size = 3
zones = [1,2,3,4,5,6]
viz.adata.obs["ploidity"] = ""

classes = {"Cell-oneNuc":("2n","4n"), "Cell-twoNuc":("2n+2n","4n+4n")}

n_rows = len(classes)
n_cols = len(zones)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=False)

def find_minimum(array, bins=50, kernel_smoothing_size=3):
    counts, bin_edges = np.histogram(array, bins=bins)
    
    # Smooth the counts
    if kernel_smoothing_size:
        counts = np.convolve(counts, np.ones(kernel_smoothing_size)/kernel_smoothing_size, mode='same')
    
    # Find peaks
    peaks, _ = find_peaks(counts)
    highest_two_peaks = peaks[np.argsort(counts[peaks])[-2:]]
    highest_two_peaks = np.sort(highest_two_peaks)
    # peak_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in highest_two_peaks]

    # Find valleys (minima) by finding peaks of the negative counts
    valleys, _ = find_peaks(-counts)
    valleys_between = valleys[(valleys > highest_two_peaks[0]) & (valleys < highest_two_peaks[1])]
    valley_values_between = [ (bin_edges[i] + bin_edges[i + 1]) / 2 for i in valleys_between]
    if len(valley_values_between) > 1:
        valley_values_between = min(valley_values_between)
    elif len(valley_values_between) == 1:
        valley_values_between = valley_values_between[0]
    else:
        valley_values_between = None
    return valley_values_between

min_vals = {}
for i, nuc in enumerate(classes):
    min_vals[nuc] = []
    for j, zone in enumerate(zones):
        ax = axes[i, j] if n_rows > 1 else axes[j]
        mask = (viz.adata.obs["zone"] == zone) & (viz.adata.obs["Classification"] == nuc)
        area = viz.adata.obs.loc[mask, "Nucleus: Area µm^2"].values
        
        min_val = find_minimum(area)
        viz.adata.obs.loc[mask & (viz.adata.obs["Nucleus: Area µm^2"] >= min_val), "ploidity"] = classes[nuc][1]
        viz.adata.obs.loc[mask & (viz.adata.obs["Nucleus: Area µm^2"] < min_val), "ploidity"] = classes[nuc][0]

        ax = sns.histplot(area,bins=bins,kde=True, ax=ax)
        ax.axvline(x=min_val, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{nuc}, zone: {zone}")
        min_vals[nuc].append(min_val)
        
plt.tight_layout()

#%%% Plot lineplots
color_map = {"2n":"red", "2n+2n":"orange", "4n":"green", "4n+4n":"blue"}
y_map = {"2n":0.65, "2n+2n":0.8, "4n":0.35, "4n+4n":0.2}

viz = both

fig, axes = plt.subplots(2, 1, figsize=(7,7))

for i, nuc in enumerate(classes):
    # keep only this class; carry sample id
    df = viz.adata.obs.loc[
        viz.adata.obs["Classification"] == nuc,
        ["source_", "zone", "ploidity"]].copy()

    # counts per sample-zone-ploidity
    cnt = (
        df.groupby(["source_","zone","ploidity"])
          .size().rename("count").reset_index()
    )
    # totals per sample-zone (to normalize within zone)
    tot = (
        df.groupby(["source_","zone"])
          .size().rename("total").reset_index()
    )

    # probability of each ploidity within a zone, per sample
    per_sample = cnt.merge(tot, on=["source_","zone"])
    per_sample["probability"] = per_sample["count"] / per_sample["total"]

    # summarize across samples: mean ± SEM
    summary = (
        per_sample.groupby(["zone","ploidity"])["probability"]
        .agg(mean="mean", std="std", n="count")     # n = #samples contributing
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])
    summary["sem"] = summary["sem"].fillna(0.0)    # handle single-sample edge cases

    ax = axes[i]
    # one line per ploidity, with SEM band
    uniq_p = list(summary["ploidity"].unique())
    label_offsets = np.linspace(0.06, -0.06, len(uniq_p)) if len(uniq_p) > 1 else [0.0]

    for j, p in enumerate(uniq_p):
        s = summary[summary["ploidity"] == p].sort_values("zone")
        c = color_map.get(p)

        ax.plot(s["zone"], s["mean"], marker="o", label=p, color=c)
        ax.fill_between(s["zone"], s["mean"] - s["sem"], s["mean"] + s["sem"],
                        alpha=0.2, linewidth=0, color=c)

        # inline colored label at the right end of the line
        x_last = s["zone"].iloc[-1]
        y_last = s["mean"].iloc[-1] + label_offsets[j]
        ax.text(x_last-0.4, y_map[p], p, color=c, ha="center", va="center",
                fontweight="bold", clip_on=False)
        

    ax.set_ylabel("Fraction")
    labels = [""] * 7; labels[1] = "Central"; labels[-1] = "Portal"
    ax.set_xticklabels(labels)
    ax.set_xlabel(None)
    ax.set_title("Mononucleated hepatocytes" if nuc=="Cell-oneNuc" else "Binucleated hepatocytes")

plt.tight_layout()
viz.plot.save("ploidity", fig=fig)

#%% Plot violins
def plot_violin(viz, y ,title, ylab, classification=None, width=0.2,color=None,umi_thresh=500,ax=None,figsize=(5,5)):
    df = viz.adata.obs.copy()
    df["zone"] = df["zone"].astype(float).astype(int).astype(str)
    zone_order = sorted(df["zone"].unique(), key=lambda x: int(x))
    df["zone"] = pd.Categorical(df["zone"], categories=zone_order, ordered=True)
    colors = cm.viridis(np.linspace(0, 1, len(zone_order)))
    color_dict = dict(zip(zone_order, colors))
    df["zone_numeric"] = df["zone"].astype(float)
    if classification:
        mask = (df["nUMI"] >= umi_thresh) & (df["Classification"]==classification )
    else:
        mask = df["nUMI"] >= umi_thresh
    df[y] = viz[y]
    df_corr = df.loc[mask].copy()
    valid = df_corr["zone_numeric"].notna() & df_corr[y].notna()
    rho, pval = spearmanr(df_corr.loc[valid, "zone_numeric"], df_corr.loc[valid, y])
    pval = pval if pval > 0 else 1e-300

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # sns.set(style="white")
    sns.violinplot(data=df_corr, x="zone", y=y, hue="zone" if color else None,
                   width=0.9,inner=None,linewidth=1, edgecolor="black", ax=ax,
                   palette = color_dict if color else None)
    sns.boxplot(data=df_corr, x="zone", y=y, width=width, showcaps=True, showfliers=False,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 3},
                whiskerprops={'color': 'black'}, capprops={'color': 'black'},
                medianprops={'color': 'black'}, zorder=3, ax=ax)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    text = f"r = {rho:.2f}, p = {pval:.2g}"
    ax.text(0.05, 0.97, text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    labels = [""] * 6; labels[0] = "Central"; labels[5] = "Portal"
    ax.set_xticklabels(labels)
    ax.set_xlabel(None)
    return ax
    

#%%% Plot
# cell size paper: https://elifesciences.org/articles/11214 

ax = plot_violin(viz, "Nucleus: Area µm^2","Mononucleated hepatocytes","Nuclear area (µm²)",
            classification="Cell-oneNuc",width=0.15)
for i, mv in enumerate(min_vals["Cell-oneNuc"]): 
    ax.hlines(mv, i - 0.4, i + 0.4, color="purple", linewidth=1, linestyle="--")
ax.set_xlim([-0.5,6.1])
ax.text(5.5, 80, "4n", color="green",fontweight="bold")
ax.text(5.5, 20, "2n", color="red",fontweight="bold")
viz.plot.save("violins1",ax=ax)

ax = plot_violin(viz, "Nucleus: Area µm^2","Binucleated hepatocytes","Nuclear area (µm²)",
            classification="Cell-twoNuc",width=0.12)
for i, mv in enumerate(min_vals["Cell-twoNuc"]): 
    ax.hlines(mv, i - 0.4, i + 0.4, color="purple", linewidth=1, linestyle="--")
ax.set_xlim([-0.5,6.5])
ax.text(5.2, 120, "4n+4n", color="blue",fontweight="bold")
ax.text(5.2, 40, "2n+2n", color="orange",fontweight="bold")
viz.plot.save("violins2",ax=ax)

ax = plot_violin(viz, "Cell: Area µm^2","Hepatocyte size","Cell area (µm²)")
viz.plot.save("violins3",ax=ax)


#%% Save
wt98.save()
wt97.save()

