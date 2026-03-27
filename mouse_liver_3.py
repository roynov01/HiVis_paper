# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:00:52 2025

@author: royno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, combine_pvalues
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
from HiVis import HiVis


#%% Import data
wt98 = HiVis.load(r"X:\roy\viziumHD\analysis\python\version_11\organs\mouse_liver\output\WT_q6\mouse_liver_98_WT.pkl")
wt97 = HiVis.load(r"X:\roy\viziumHD\analysis\python\version_11\organs\mouse_liver\output\WT_q6\mouse_liver_97_WT.pkl")


#%% define BV
for viz in [wt98,wt97]:
    # mask = (viz["blood_vessels_fullres: Blood_vessel %"] > 90) & ~viz["InCell"].astype(bool)
    mask = (viz["blood_vessels_fullres: Blood_vessel %"] > 90) 

    viz["BV"] = mask.astype(str)
    viz.adata.obs.loc[viz.adata.obs["BV"] == "False","BV"] = np.nan
    viz.adata.obs.loc[viz.adata.obs["Classification"]=="BV-Cell","BV"] = "True"
    viz["DistToCell"] = abs(viz.adata.obs["DistToCell"])
    
#%% plot spatial & distances

xlim = [25,225]
ylim = [500,600]

t = wt98[(wt98["um_x"] > xlim[0]) & (wt98["um_x"] < xlim[1]) &
         (wt98["um_y"] > ylim[0]) & (wt98["um_y"] < ylim[1]),:]
t.recolor(fluorescence={'ATP1A1': 'red', 'CD31': "green", 'autofluorescence': None, 'DAPI': 'blue'},
          normalization_method = "percentile")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,7))
t.plot.spatial(ax=axes[0],axis_labels=False,title="Mouse liver")

ax = t.agg["HEP"].plot.cells(line_color="cyan",ax=axes[1],scalebar=False)
ax = t.plot.spatial("BV",cmap=["green","green"],ax=axes[1],legend=False,title="Pixel classifier - sinusoids",
               scalebar={"text":False})
bv_patch = mpatches.Patch(color='green', label='Sinusoid')
bv_patch2 = mpatches.Patch(color='cyan', label='Cell border')
ax.legend(handles=[bv_patch,bv_patch2], title=None, loc='upper right')

plt.tight_layout()
wt98.plot.save("spatial_BV", fig=fig)
#%%% distances
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7,7))
ax = wt98.agg["HEP"].plot.cells(xlim=xlim,ylim=ylim,image=False,ax=axes[0],line_color="k",
                                axis_labels=False, scalebar=False)
wt98.plot.spatial("BV",xlim=xlim,ylim=ylim,exact=True,image=False,scalebar=False,
                  cmap=["green","green"],ax=ax,legend=False,axis_labels=False)
wt98["temp"] = wt98["dist_to_bv_um"]; wt98.adata.obs.loc[~wt98["InCell"].astype(bool),"temp"] = np.nan
wt98.plot.spatial("temp",xlim=xlim,ylim=ylim,exact=True, legend_title="",
                  # scalebar={"bar_offset":0.04,"text_offset":0.08},
                  alpha=1,ax=ax,image=False,cmap="autumn",axis_labels=False,title="Distance to sinusoid")
bv_patch = mpatches.Patch(color='green', label='Sinusoid')
ax.legend(handles=[bv_patch], title=None, loc='upper right')

ax = wt98.agg["HEP"].plot.cells(xlim=xlim,ylim=ylim,image=False,ax=axes[1],line_color="k",
                                axis_labels=False, scalebar=False)
wt98.plot.spatial("BV",xlim=xlim,ylim=ylim,exact=True,image=False, scalebar=False,
                  cmap=["green","green"],ax=ax,legend=False,axis_labels=False)
wt98.plot.spatial("DistToCell",xlim=xlim,ylim=ylim,exact=True, legend_title="",
                  # scalebar={"bar_offset":0.04,"text_offset":0.08},
                  alpha=1,ax=ax,image=False,cmap="autumn",axis_labels=False,title="Distance to cell border")
bv_patch = mpatches.Patch(color='green', label='Sinusoid')
ax.legend(handles=[bv_patch], title=None, loc='upper right')

plt.tight_layout()
wt98.plot.save("distances", fig=fig)

#%% Assign apicome and nuc/cyto

basal_thresh = 1.5
cortical_thresh = 1.5

for viz in [wt98,wt97]:
    viz.adata.obs["apicome"] = viz.adata.obs["BV"].copy()

    
    # Assign identity for each bin
    viz.adata.obs.loc[viz.adata.obs["apicome"] == "True","apicome"] = "BV" 
    viz.adata.obs.loc[viz.adata.obs["dist_to_bv_um"] == 0,"apicome"] = "BV" 
    in_hepato = viz.adata.obs["Cell_ID"].isin(viz.agg["HEP"].adata.obs.index) & ~(viz.adata.obs["apicome"]=="BV")
    
    viz.adata.obs.loc[in_hepato & 
                      (viz.adata.obs["DistToCell"] <= cortical_thresh),"apicome"] = "cortical"
    viz.adata.obs.loc[in_hepato & 
                      (viz.adata.obs["DistToCell"] <= cortical_thresh) &
                      (viz.adata.obs["dist_to_bv_um"] <= basal_thresh), "apicome"] = "basal"
    viz.adata.obs.loc[in_hepato &
                      (viz.adata.obs["InNuc"] == 1) ,"apicome"] = "nuc"
    viz.adata.obs.loc[in_hepato & 
                      ~viz.adata.obs["apicome"].isin(["cortical","basal","nuc"]), "apicome"] = "cyto"
#%%% Plot spatial
xlim = [25,225]
ylim = [400,600]
# ylim = [500,600]

wt98["temp"] = wt98.adata.obs["apicome"].str.capitalize()
wt98.update_meta("temp",{"Bv":"Sinusoid","Cortical":"Cortical","Nuc":"Nuclear","Cyto":"Core"})
ax = wt98.plot.spatial("temp",xlim=xlim,ylim=ylim,legend=True,exact=True,
                       cmap={"Basal":"blue","Nuclear":"cyan","Cortical":"red","Sinusoid":"green",
                             "Core":"#f7ae54"},image=False,scalebar=False,
                       title="Bins colored by identity",legend_title="")

ax = wt98.agg["HEP"].plot.cells(xlim=xlim,ylim=ylim,image=False,ax=ax,line_color="black")
ax.get_legend().set_loc('upper right')
ax.get_legend().set_bbox_to_anchor((0.98, 0.98)) 
wt98.plot.save("APICOME_SPATIAL", ax=ax)

#%% import signature-matrix of cells in liver (Ben-Moshe et al. 2022)
bm = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\input\benMoshe_2022_apap_0h.csv",index_col=0)
hepato_name = "Hep"
stromal = bm[[col for col in bm.columns if not col.startswith(hepato_name)]]
stromal = stromal.mean(axis=1)
hep = bm[hepato_name]
pn = hep[hep>0].min()
ratio = (hep+pn) / (stromal+pn)

ratio_thresh = 1
exp_thresh = 1e-5
mito_genes = ratio[ratio.index.str.startswith("mt-")]

hep_markers = bm.index[(ratio >= ratio_thresh) & (bm[hepato_name] > exp_thresh) &
                       ~ratio.index.str.startswith("mt-")]

hep_markers = hep_markers.intersection(wt98.adata.var_names)
wt98_hep = wt98[:,hep_markers]
wt98_hep.properties["hep_only"], wt98.properties["hep_only"] = True, False

hep_markers = hep_markers.intersection(wt97.adata.var_names)
wt97_hep = wt97[:,hep_markers]
wt97_hep.properties["hep_only"], wt97.properties["hep_only"] = True, False
#%% DGE

zones_sep = [[1,2],[5,6],[1,2,3,4,5,6]]
zones_names = "central", "portal","allzones"

umi_thresh = 500

for viz in [wt98_hep,wt97_hep]:
    import gc;gc.collect()
    high_exp_cells = viz.agg["HEP"].adata.obs.index[(viz.agg["HEP"].adata.obs["nUMI"] >= umi_thresh)]
    
    basal_bins = viz.adata.obs.loc[(viz.adata.obs["apicome"] == "basal")&
                                   (viz.adata.obs["InCell"] == 1) &
                                   (viz.adata.obs["Classification"] != "BV-Cell")]
    basal_counts = basal_bins.groupby("Cell_ID").size()
    cell_ids_with_many_basal = basal_counts[basal_counts >= 3].index.tolist() 
    
    cells_without_BV_inside = viz.agg["HEP"]["blood_vessels_fullres: Blood_vessel %"] < 20
        
    # valid_cells_for_apicome = viz.agg["HEP"].adata.obs.loc[cell_ids_with_many_basal].index
    all_cells = viz.agg["HEP"].adata.obs.index
    valid_cells_for_apicome = viz.agg["HEP"].adata.obs.loc[all_cells.isin(cell_ids_with_many_basal) &
                                                           cells_without_BV_inside].index
    
    valid_hepatocyte = high_exp_cells.intersection(valid_cells_for_apicome)
    viz.properties["valid_hepatocyte"] = valid_hepatocyte
    retentions, apicomes = [], []
    for name, sep in zip(zones_names,zones_sep):
        print(f"Running: {viz.name}, {name}")
        sub = viz[viz.adata.obs["zone"].isin(sep) &
                  viz.adata.obs["Cell_ID"].isin(valid_hepatocyte),:]
        
        # Apicome
        apicome = sub.analysis.dge("apicome", group1="cortical", group2="basal",umi_thresh=10,
                        method="fisher_exact",two_sided=False,inplace=False)
        apicome.columns = [f"{col}_{name}" for col in apicome.columns]
        apicomes.append(apicome)
        
        # Retention
        sub = viz[viz.adata.obs["zone"].isin(sep) &
                  viz.adata.obs["Cell_ID"].isin(high_exp_cells),:]

        retention = sub.analysis.dge("apicome", group1="nuc", group2="cyto",umi_thresh=10,
                        method="fisher_exact",two_sided=False,inplace=False)
        retention.columns = [f"{col}_{name}" for col in retention.columns]
        retentions.append(retention)

    retention_combined = pd.concat(retentions, axis=1)
    filename = f"retention{'_epi_genes' if viz.properties['hep_only'] else ''}"
    viz.plot.save(filename,fig=retention_combined)
    viz.properties["retention"] = retention_combined
    apicomes_combined = pd.concat(apicomes, axis=1)
    filename = f"apicome{'_epi_genes' if viz.properties['hep_only'] else ''}"
    viz.plot.save(filename,fig=apicomes_combined)   
    viz.properties["apicome"] = apicomes_combined

import gc;gc.collect()

#%% Plot FACS plots
def add_square(ax, y_max, x_max, label, x_min=0, y_min=0,x_text=None,
               x_margin=2.5, y_margin=0.2, color="red",color_text=None):
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                fill=False, edgecolor=color, linewidth=2, linestyle="--"))
    ax.text(x_text if x_text is not None else x_max - x_margin, y_max + y_margin, label,
                color=color if color_text is None else color_text, 
                ha="center", va="bottom")
    return ax

for viz in [wt98_hep,wt97_hep]:
    sub = viz[viz.adata.obs["Cell_ID"].isin(viz.properties["valid_hepatocyte"]) &
              viz.adata.obs["dist_to_bv_um"] > 0,:]

    counts = sub.adata.obs["apicome"].value_counts(normalize=True) * 100
    
    ax, info = HiVis.HiVis_plot.plot_density(sub,x="dist_to_bv_um",y="DistToCell",count="apicome",
                                             cmap="turbo",figsize=(6,6),legend_title="log10(bin density)",
                         gridsize=300, mincnt=30,xlab="Distance to sinusoid (µm)",
                         ylab="Distance to cell border (µm)",title="Basal/cortical classification")
    
    ax.vlines(x=basal_thresh,  ymin=0,ymax=cortical_thresh, color="k", linestyle="--",linewidth=2)
    ax.axhline(y=basal_thresh, color="k", linestyle="--",linewidth=2)
    ax.text(basal_thresh-0.75,basal_thresh-1.05, f"Basal\n{counts['basal']:.1f}%",color="k",ha="center", va="bottom")
    ax.text(cortical_thresh+0.75,basal_thresh-1.05, f"Cortical\n{counts['cortical']:.1f}%",color="k",ha="center", va="bottom")

    vmin, vmax = info["cbar"].norm.vmin, info["cbar"].norm.vmax
    ticks = sorted(set([100, int(vmin), int(round(vmax, -2)) ]))
    info["cbar"].set_ticks(ticks)
    info["cbar"].set_ticklabels(ticks)
    viz.plot.save("FACS_apicome",ax=ax) 
    break


import gc;gc.collect()
#%% Plot MAs
qval_thresh_a = 0.01
qval_thresh_r = 1e-5

exp_thresh = 1e-5

for viz in [wt98_hep,wt97_hep]:
    plot_a, plot_r = viz.properties["apicome"].copy(), viz.properties["retention"].copy()
    plot_a["gene"], plot_r["gene"] = plot_a.index, plot_r.index
    for name, sep in zip(zones_names,zones_sep):
        title = f"WT{viz.name.split('_')[2]},   Apicome{',   '+name}{',   epithel genes' if viz.properties['hep_only'] else ''}"
        ax = HiVis.HiVis_plot.plot_MA(plot_a, qval_thresh=qval_thresh_a, exp_thresh=exp_thresh,
                              title=title,
                              colname_exp=f"expression_min_{name}", colname_qval=f"qval_{name}", 
                              colname_fc=f"log2fc_{name}", ylab="log2(cortical/basal)",n_texts=250,repel=True)

        viz.plot.save(f"apicome_{name}_MA",ax=ax)
        title = f"WT{viz.name.split('_')[2]},   Retention{',   '+name}{',   epithel genes' if viz.properties['hep_only'] else ''}"
        ax = HiVis.HiVis_plot.plot_MA(plot_r, qval_thresh=qval_thresh_r, exp_thresh=exp_thresh,
                              title=title,
                              colname_exp=f"expression_min_{name}", colname_qval=f"qval_{name}", 
                              colname_fc=f"log2fc_{name}", ylab="log2(nucleus/cytoplasm)",n_texts=250,repel=True)
        viz.plot.save(f"retention_{name}_MA",ax=ax)
        plt.show()
    
#%% Export pseudobulk apicome & retention

for viz in [wt98,wt97,wt98_hep,wt97_hep]:
    pb = viz.analysis.pseudobulk("apicome")
    viz.plot.save(f"apicome_pb{'_epi' if viz.properties['hep_only'] else ''}",fig=pb)


#%% Merge both samples apicome
qval_thresh = 0.25
exp_thresh = 1e-5

reducer = lambda row: combine_pvalues(row, method="fisher",nan_policy="omit")[1]

df1 = wt98_hep.properties["apicome"]
df2 = wt97_hep.properties["apicome"]
apicomes_combined = {}
group_names = ["cortical","basal"]

for name in zones_names:
    df1_cur = df1[[col for col in df1.columns if col.endswith(f"_{name}")]]
    df1_cur.columns = [col.replace(f"_{name}","") for col in df1_cur.columns]
    df2_cur = df2[[col for col in df2.columns if col.endswith(f"_{name}")]]
    df2_cur.columns = [col.replace(f"_{name}","") for col in df2_cur.columns]

    df = HiVis.HiVis_utils.combine_dges([df1_cur,df2_cur],group_names,pval_reducer=reducer,
                        log2fc_reducer=np.nanmedian, expression_reducer=np.nanmean, exp_thresh=exp_thresh)

    apicomes_combined[name] = df


#%% Plot MA apicome

qval_thresh = 0.25
exp_thresh = 1e-5

basal_examples = ["Cyb5r3","Pigr"]

df_cur = apicomes_combined["allzones"]

plot = df_cur.loc[df_cur["expression_max"] >= exp_thresh].copy()
plot["expression_min"] = np.log10(plot["expression_min"])
plot["count_max"] = plot[["count_cortical","count_basal"]].max(axis=1)


apical_plot = plot.loc[plot["log2fc"] >= 0]
basal_plot = plot.loc[plot["log2fc"] < 0]

apical_genes = apical_plot.index[(apical_plot["qval"] < qval_thresh) & (apical_plot["count_max"] > 1)]
basal_genes = basal_plot.index[(basal_plot["qval"] < qval_thresh) & (basal_plot["count_max"] > 1)]
basal_genes = [b for b in basal_genes if b not in basal_examples]
signif = plot.index[(plot["qval"] < qval_thresh) & (plot["count_max"] > 4)] # two patients is less than half


ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=apical_genes,color_genes="red",
                                          genes2=basal_genes,color_genes2="blue",repel=True,
                                          text=True, color="gray",figsize=(6,6))



plot2 = plot.loc[plot["gene"].isin(basal_examples)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_min","log2fc",
                                          xlab="log10(expression)", bold=True,
                                          title="Polarized genes - mouse liver",ylab="log2(cortical/basal)",
                                          ax=ax,size=35,color_genes="blue",repel=True,
                                          genes=basal_examples, y_line=0)
ax.text(0.98, 0.95,"Cortical mRNAs",transform=ax.transAxes,ha='right',color="red",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Basal mRNAs",transform=ax.transAxes,ha='right',color="blue",
        bbox=dict(facecolor='white', edgecolor='black'))

wt98.plot.save("apicome_MA",ax=ax)


#%% Merge both samples retention
qval_thresh = 0.25
exp_thresh = 1e-5

reducer = lambda row: combine_pvalues(row, method="fisher",nan_policy="omit")[1]

df1 = wt98_hep.properties["retention"]
df2 = wt97_hep.properties["retention"]
retentions_combined = {}

group_names = ["nuc","cyto"]

for name in zones_names:
    df1_cur = df1[[col for col in df1.columns if col.endswith(f"_{name}")]]
    df1_cur.columns = [col.replace(f"_{name}","") for col in df1_cur.columns]
    df2_cur = df2[[col for col in df2.columns if col.endswith(f"_{name}")]]
    df2_cur.columns = [col.replace(f"_{name}","") for col in df2_cur.columns]

    df = HiVis.HiVis_utils.combine_dges([df1_cur,df2_cur],group_names,pval_reducer=reducer,
                        log2fc_reducer=np.nanmedian, expression_reducer=np.nanmean, exp_thresh=exp_thresh)

    retentions_combined[name] = df


#%% Plot MA retention
qval_thresh = 0.01
exp_thresh = 1e-5

df_cur = retentions_combined["allzones"]

plot = df_cur.loc[df_cur["expression_max"] >= exp_thresh].copy()

plot["count_max"] = plot[["count_nuc","count_cyto"]].max(axis=1)

signif = plot.index[(plot["qval"] < qval_thresh) & (plot["count_max"]>1)]

plot["expression_min"] = np.log10(plot["expression_min"])

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_min", 
                                          "log2fc",genes=signif,
                                          text=False, color="gray",figsize=(6,6))

most_polarized = ["Mlxipl", "Sema4g", "Tat", "Col27a1", "Pck1", "Ces3a","Nlrp6","Itih3","Nr1i3",
                  "Scd1","Apoe","Alb","Pigr","Apoc1","Apoa2"]
# most_polarized = plot.index[plot["qval"] < 1e-200]
plot2 = plot.loc[plot["gene"].isin(most_polarized)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_min",
                                          "log2fc",genes=most_polarized, ax=ax,repel=True,
                                          xlab="log10(expression)", 
                                          title="Cytoplasmic bias - mouse liver",
                                          ylab="log2(nucleus/cytoplasm)",figsize=(6,6),
                                          text=True, color="gray",y_line=0)
ax.text(0.95, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="limegreen")
ax.text(0.95, 0.05,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="limegreen")


wt98.plot.save("retention_MA",ax=ax)

#%% Plot MA retention (new)
# df_cur = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_retention_allzones.csv",index_col=0)


qval_thresh = 0.01
exp_thresh = 1e-5

plot = df_cur.loc[df_cur["expression_mean"] >= exp_thresh].copy()

nuc_examples = ["Mlxipl","Sema4g","Nr1i3","Nlrp6","Itih3","Tat","Ces3a","Col27a1","Pck1"]
cyto_examples = ["Pigr","Scd1","Apoc1","Apoa2","Apoe","Alb"]


plot["expression_mean"] = np.log10(plot["expression_mean"])
plot["count_max"] = plot[["count_nuc","count_cyto"]].max(axis=1)

nuc_plot = plot.loc[plot["log2fc"] >= 0]
cyto_plot = plot.loc[plot["log2fc"] < 0]

nuc_genes = nuc_plot.index[(nuc_plot["qval"] < qval_thresh)& (nuc_plot["count_max"]>1)]
nuc_genes_text = nuc_plot.index[(nuc_plot["qval"] < qval_thresh) & nuc_plot.index.isin(nuc_examples) & (nuc_plot["count_max"]>1)]


cyto_genes = cyto_plot.index[(cyto_plot["qval"] < qval_thresh)& (cyto_plot["count_max"]>1)]
cyto_genes_text = cyto_plot.index[(cyto_plot["qval"] < qval_thresh) & cyto_plot.index.isin(cyto_examples)& (cyto_plot["count_max"]>1) ]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", 
                                          "log2fc",genes=nuc_genes_text,color_genes="darkcyan",
                                          genes2=cyto_genes_text,color_genes2="darkorange",repel=1,
                                          text=True, color="gray",y_line=0,figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", "log2fc",
                                          genes=nuc_genes,color_genes="darkcyan",
                                          genes2=cyto_genes,color_genes2="darkorange",
                                          text=False, color="gray",ax=ax,
                                          xlab="log10(expression)",ylab="log2(nucleus/cytoplasm)",
                                          title="Cytoplasmic bias - mouse liver")

ax.text(0.98, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="darkcyan",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="darkorange",
        bbox=dict(facecolor='white', edgecolor='black'))


plt.savefig(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_retention_MA.pdf")
plt.savefig(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_retention_MA.png")
# wt98.plot.save("MA_retention",ax=ax)


#%% Retention across zones
qval_thresh = 0.25
exp_thresh = 1e-4

df1 = retentions_combined["central"].rename(columns={c: f"{c}_c" for c in retentions_combined["central"].columns if c != "gene"})
df2 = retentions_combined["portal"].rename(columns={c: f"{c}_p" for c in retentions_combined["portal"].columns if c != "gene"})

merged = df1.merge(df2, on="gene", how="outer")

plot = merged.loc[(merged["expression_max_c"] >= exp_thresh) & (merged["expression_max_p"] >= exp_thresh)&
                  (merged["qval_c"] <= qval_thresh) & (merged["qval_p"] <= qval_thresh)]
plot["qval"] = plot[["qval_p","qval_c"]].max(axis=1)

genes = ["Mlxipl","Sema4g","Vegfa","Pck1", "Nlrp6","Itih3","Apoc1","Pigr","Scd1","Entpd5","Gstt2","Ubr3","Psmd3"]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "log2fc_c", "log2fc_p",
                                  xlab="log2(nucleus/cytoplasm) - central",title="Cytoplasmic bias across zones",
                                  ylab="log2(nucleus/cytoplasm) - portal",color_genes="black",
                                  genes=genes,text=True,
                                  color="black",x_line=0,y_line=0,figsize=(6,6),repel=True)
corr, pval = spearmanr(plot["log2fc_c"], plot["log2fc_p"])
pval = pval if pval > 0 else 1e-300

ax.text(0.6, 0.05,f"r = {corr:.2f}, p = {pval:.2g}",transform=ax.transAxes)
wt98.plot.save("retention_across_zones",ax=ax)



#%%  Correlation with Bahar et al 2015
path = r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\input\bahar_2015.csv"
pn = 1
bahar = pd.read_csv(path, index_col=0).filter(like='liver')
bahar["nuc"] = (bahar["Nuc liver 1"] + bahar["Nuc liver 2"]) / 2
bahar["cyto"] = (bahar["Cyto liver 1"] + bahar["Cyto liver 2"]) / 2
bahar["ratio liver"] = (bahar["nuc"] + pn) / (bahar["cyto"] + pn)
bahar["ratio liver"] = np.log2(bahar["ratio liver"])
bahar["expression liver"] = bahar[["nuc","cyto"]].max(axis=1)

merged = pd.merge(retentions_combined["allzones"],bahar , left_index=True, right_index=True)
merged['gene'] = merged.index

exp_thresh = 1e-4
qval_thresh = 0.25
plot = merged.loc[(merged['expression liver'] >= exp_thresh) &(merged['qval'] <= qval_thresh)&
                   (merged['expression_min'] >= exp_thresh)].copy()

plot = plot.dropna()

# genes = list(plot.index[(plot["ratio liver"] > 0) | 
#                         (plot["log2fc"] > 0.6) | (plot["log2fc"] < -0.3)])
genes = list(plot.index)
genes = ["Mlxipl","Ccnl2","Vegfa", "Nlrp6","Gcgr","Fth1","Pigr","Cyb5r3"]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "ratio liver", "log2fc",genes=genes,
                                  xlab="log2(nuc/cyto) - fractions RNAseq",text=True,repel=True,
                                  ylab="log2(nuc/cyto) - VisiumHD",color_genes="black",
                                  color="black",figsize=(6,6))
corr, pval = spearmanr(plot["ratio liver"], plot["log2fc"])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}", transform=ax.transAxes)
ax.set_title("Correlation with Bahar Halpern et al. (2015)")
wt98.plot.save("retention_correlation",ax=ax)

#%% Export
wt98.plot.save("retention_allzones",fig=retentions_combined["allzones"])
wt98.plot.save("retention_central",fig=retentions_combined["central"])
wt98.plot.save("retention_portal",fig=retentions_combined["portal"])
wt98.plot.save("apicome",fig=apicomes_combined["allzones"])
#%%%
if 0:
    for i, viz in enumerate([wt97.copy(),wt98.copy()]):
        viz.rename(f"liver_rep{i+1}",new_out_path=False, full=True)
        _ = viz.export_images()
        _ = viz.export_h5()
        viz.agg["SC"].rename("single_cells")
        _ = viz.agg["SC"].export_h5()
        viz.agg["HEP"].rename("hepatocytes")
        _ = viz.agg["HEP"].export_h5()
        del viz.agg["temp"]
        del viz.agg["nuc"]
        viz.save()

