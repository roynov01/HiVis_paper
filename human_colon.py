# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 12:26:20 2025

@author: royno
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, combine_pvalues
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

path_input = r"X:\roy\viziumHD\data\human_colon"
path_output = r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_colon\output_new"

#%% Set properties
patients = ["P1","P2","P3","P5","P5N"]
cancers = [True, True, False, True, False]

organism, organ, source = "human","colon","10X"

export = False # export .pkl files?
_import = True # True to import .pkl files. otherwise will create new instances of ViziumHD

#%% Create or import Viziums
viziums = []
 
for i, patient in enumerate(patients):
    name = f"oliviera_{patient}"
    if _import:
        print(f"**[loading] {name}**")
        li = HiVis.load(name,directory=f"{path_output}/{patient}")
        li.update()
    else:
        print(f"**[creating] {name}**")
        properties = {"organism":organism,
                      "organ":organ,
                      "patient": patient,
                      "cancer": cancers[i],
                      "sample_id":name,
                      "source":source}
        path_input_cur = f"{path_input}/{patient}"
        path_input_fullres_image = f"{path_input_cur}/{patient}.btf"
        path_input_data = path_input_cur + "/square_002um"
        path_output_cur =  f"{path_output}/{patient}"
        li = HiVis.new(path_input_fullres_image, path_input_data, 
                                     path_output_cur, name,  properties=properties,
                                     on_tissue_only=True,min_reads_in_spot=1,
                                     min_reads_gene=10)
    plt.show()
    viziums.append(li)

print("Set xlims, ylims for next stage!")


#%% Add apicome annotations
condition_name = "apicome"
xlims = [
    [800,1600],
    [4500,5300],
    [970,1170],
    [1200,1999],
    [3000,3500]
    ]
ylims = [
    [5400,6100],
    [4600,5400],
    [2750,2950],
    [2200,2999],
    [2200,2999]
    ]
xlims = {patient: xlim for patient, xlim in zip(patients, xlims)}
ylims = {patient: ylim for patient, ylim in zip(patients, ylims)}

if not _import:
    for i, patient in enumerate(patients):
        li = viziums[i]
        
        # import apicome annotations
        annotations_path = fr"X:\roy\viziumHD\analysis\Qupath\human_colon\export/{patient}_cropped - apicome.geojson"
        li.add_annotations(annotations_path,condition_name)
        c = {"apical":"red","nuc": "orange","basal":"green"}
        li.plot.hist(condition_name,cmap=c,save=True)
        xlim_selected, ylim_selected = xlims[patient], ylims[patient]
        li.plot.spatial(condition_name,cmap=c,xlim=xlim_selected,
                        ylim=ylim_selected,alpha=0.2,save=True)
        
        # move the "nuc" classification from "apicome":
        if not "nuc_cyto" in li.adata.obs.columns: 
            li.adata.obs["nuc_cyto"] = li.adata.obs["apicome"].copy()
            li.update_meta(name="nuc_cyto", values={"apical": "cyto", "basal": "cyto"})
            li.update_meta(name="apicome", values={"nuc": np.nan})
        li.plot.spatial("nuc_cyto",xlim=xlim_selected,
                        ylim=ylim_selected,alpha=0.2,save=True)
  
#%% export
if export:
    for li in viziums:
        path = li.save()
#%% Apicome
#%%% Enterocyte specific genes
expression = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_colon\input\human_colon_signature.csv") 
expression.drop(["Tuft", "stem.TA", "Enteroendocrine", "Best4..Enterocytes", "WNT2B..RSPO3", "WNT5B"], inplace=True, axis=1, errors='ignore')
expression["DC"] = expression[["DC1","DC2"]].mean(axis=1)
epi_genes, _ =  HiVis.other_utils.find_markers(expression,
                            celltypes=["Enterocyte","Goblet"],
                            ratio_thresh=1,
                            exp_thresh=1e-5,
                            chosen_fun="max",
                            other_fun="mean")
mito_genes = [i for i in epi_genes if i.startswith("MT-")]
#%%% DGE (per patient)
use_epi_genes = True

dataframes = [None for _ in patients]
for i, patient in enumerate(patients):
    # filter epithelial genes only:
    li = viziums[i]
    if use_epi_genes:
        li = li[:,li.adata.var.index.isin(epi_genes) & ~li.adata.var.index.isin(mito_genes)]

    df = li.analysis.dge("apicome", group1="apical", group2="basal",method="fisher_exact",
                         umi_thresh=10,two_sided=False)
    
    q_thresh = 0.01
    exp_thresh = 1e-5
    
    ax = HiVis.HiVis_plot.plot_MA(df, qval_thresh=q_thresh, 
                               exp_thresh=exp_thresh, 
                          title=f"{patient}, Q < {q_thresh}",
                          colname_exp="expression_mean", colname_qval="qval", 
                          colname_fc="log2fc", ylab="log2(apical/basal)")
    li.plot.save(f"apicome_MA{'_epi_genes' if use_epi_genes else '_all_genes'}",ax=ax)
    li.plot.save(f"apicome{'_epi_genes' if use_epi_genes else '_all_genes'}",fig=df)

    # df.columns = [f"{col}_{patient}" for co2l in df.columns]
    dataframes[i] = df
#%%% Combine patients
exp_thresh = 0
reducer = lambda row: combine_pvalues(row, method="pearson",nan_policy="omit")[1]

combined_apicome = HiVis.HiVis_utils.combine_dges(dataframes,["apical","basal"],pval_reducer=reducer,
                    log2fc_reducer=np.nanmedian, expression_reducer=np.nanmean, exp_thresh=exp_thresh)
combined_apicome.to_csv(f"{path_output}/combined_apicome_epi_genes.csv")

qval_thresh = 0.25
exp_thresh = 1e-5
exp_thresh_extreme = 1e-80

apical_examples = ["PIGR"]
basal_examples = ["MYH14"]


plot = combined_apicome.loc[combined_apicome["expression_mean"] >= exp_thresh].copy()
plot["expression_mean"] = np.log10(plot["expression_mean"])
plot["count_max"] = plot[["count_apical","count_basal"]].max(axis=1)


apical_plot = plot.loc[plot["log2fc"] >= 0]
basal_plot = plot.loc[plot["log2fc"] < 0]

apical_genes = apical_plot.index[(apical_plot["qval"] < qval_thresh) & (apical_plot["count_max"] > 4)]
apical_genes_text = ["MUC12","CEACAM5","LGALS3BP","HOOK1","CDH1"]
basal_genes = basal_plot.index[(basal_plot["qval"] < qval_thresh) & (basal_plot["count_max"] > 4)]
basal_genes_text = ["KTN1","NET1","KIF1C","DST","DSP","NFE2L1"]

signif = plot.index[(plot["qval"] < qval_thresh) & (plot["count_max"] > 4)] # two patients is less than half


ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", 
                                          "log2fc",genes=apical_genes_text,color_genes="red",
                                          genes2=basal_genes_text,color_genes2="blue",repel=True,
                                          text=True, color="gray",figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", 
                                          "log2fc",genes=apical_genes,color_genes="red",
                                          genes2=basal_genes,color_genes2="blue",
                                          text=False, color="gray",ax=ax)


plot2 = plot.loc[plot["gene"].isin(apical_examples)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_mean","log2fc",
                                          ax=ax,size=35,color_genes="red",repel=True,bold=True,
                                          genes=apical_examples)
plot2 = plot.loc[plot["gene"].isin(basal_examples)]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot2, "expression_mean","log2fc",
                                          xlab="log10(expression)", bold=True,
                                          title="Polarized genes - human colon",ylab="log2(apical/basal)",
                                          ax=ax,size=35,color_genes="blue",repel=True,y_line=0,
                                          genes=basal_examples)
ax.text(0.98, 0.95,"Apical mRNAs",transform=ax.transAxes,ha='right',color="red",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Basal mRNAs",transform=ax.transAxes,ha='right',color="blue",
        bbox=dict(facecolor='white', edgecolor='black'))

plt.savefig(f"{path_output}/combined_MA_epi_genes.pdf",bbox_inches='tight',pad_inches=0)
plt.savefig(f"{path_output}/combined_MA_epi_genes.svg",bbox_inches='tight',pad_inches=0)
plt.savefig(f"{path_output}/combined_MA_epi_genes.png",bbox_inches='tight',pad_inches=0)


#%% Retention
#%%% DGE (per patient)
use_epi_genes = True
retentions = [None for _ in patients]
for i,patient in enumerate(patients):
    li = viziums[i]
    if epi_genes:
        li = li[:,li.adata.var.index.isin(epi_genes) & ~li.adata.var.index.isin(mito_genes)]
    
    df = li.analysis.dge("nuc_cyto", group1="nuc", group2="cyto",method="fisher_exact",
                         umi_thresh=10,two_sided=False)
    
    q_thresh = 0.01
    exp_thresh = 1e-5
    
    ax = HiVis.HiVis_plot.plot_MA(df, qval_thresh=q_thresh, 
                               exp_thresh=exp_thresh, 
                          title=f"retention, {patient}, Q < {q_thresh}",
                          colname_exp="expression_mean", colname_qval="qval", 
                          colname_fc="log2fc", ylab="log2(nuc/cyto)")
    li.plot.save(f"retention_MA{'_epi_genes' if epi_genes else '_all_genes'}",ax=ax)
    li.plot.save(f"retention{'_epi_genes' if epi_genes else '_all_genes'}",fig=df)
    
    # df.columns = [f"{col}_{patient}" for col in df.columns]
    retentions[i] = df
#%%% Combine patients
# combined_retention = combine_summaries(retentions)
# combined_retention["qval"] = HiVis.HiVis_utils.p_adjust(combined_retention["pval"])
# combined_retention.to_csv(f"{path_output}/combined_retention_epi_genes.csv")
reducer = lambda row: combine_pvalues(row, method="fisher",nan_policy="omit")[1]

combined_retention = HiVis.HiVis_utils.combine_dges(retentions,["nuc","cyto"],pval_reducer=reducer,
                    log2fc_reducer=np.nanmedian, expression_reducer=np.nanmean, exp_thresh=exp_thresh)
combined_retention.to_csv(f"{path_output}/combined_retention_epi_genes.csv")

# combined_retention = pd.read_csv(f"{path_output}/combined_retention_epi_genes.csv",index_col=0)

#%%% MA (new)
qval_thresh = 0.05
exp_thresh = 1e-5

plot = combined_retention.loc[combined_retention["expression_mean"] >= exp_thresh].copy()
plot["expression_mean"] = np.log10(plot["expression_mean"])

plot["count_max"] = plot[["count_nuc","count_cyto"]].max(axis=1)
# signif = plot.index[(plot["qval"] < qval_thresh) & (plot["count_max"] > 4)] # two patients is less than half


nuc_examples = ["MYO15B","SLC26A2","CYP3A5","CLDN4","VMP1","TSPAN1"]
cyto_examples = ["SLC40A1","PIGR","FTH1","FABP1","ZG16"]


nuc_plot = plot.loc[plot["log2fc"] >= 0]
cyto_plot = plot.loc[plot["log2fc"] < 0]

nuc_genes = nuc_plot.index[(nuc_plot["qval"] < qval_thresh) & (nuc_plot["count_max"] > 4)]
nuc_genes_text = nuc_plot.index[(nuc_plot.index.isin(nuc_genes)) & nuc_plot.index.isin(nuc_examples) ]


cyto_genes = cyto_plot.index[(cyto_plot["qval"] < qval_thresh) & (cyto_plot["count_max"] > 4)]
cyto_genes_text = cyto_plot.index[(cyto_plot.index.isin(cyto_genes)) & cyto_plot.index.isin(cyto_examples) ]

ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", 
                                          "log2fc",genes=nuc_genes_text,color_genes="darkcyan",
                                          genes2=cyto_genes_text,color_genes2="darkorange",repel=1,
                                          text=True, color="gray",y_line=0,figsize=(6,6))
ax = HiVis.HiVis_plot.plot_scatter_signif(plot, "expression_mean", "log2fc",
                                          genes=nuc_genes,color_genes="darkcyan",
                                          genes2=cyto_genes,color_genes2="darkorange",
                                          text=False, color="gray",ax=ax,
                                          xlab="log10(expression)",ylab="log2(nucleus/cytoplasm)",
                                          title="Cytoplasmic bias - human colon")

ax.text(0.98, 0.95,"Nuclear mRNAs",transform=ax.transAxes,ha='right',color="darkcyan",
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.98, 0.03,"Cytoplasmic mRNAs",transform=ax.transAxes,ha='right',color="darkorange",
        bbox=dict(facecolor='white', edgecolor='black'))

plt.savefig(f"{path_output}/combined_MA_retention_epi_genes.pdf",bbox_inches='tight',pad_inches=0,dpi=300)
plt.savefig(f"{path_output}/combined_MA_retention_epi_genes.png",bbox_inches='tight',pad_inches=0,dpi=300)

#%% Plot spatial
patients_dict = dict(zip(patients,viziums))

# for i, patient in enumerate(patients):
#     li = viziums[i]
#     li.plot.spatial(xlim=xlims[patient],ylim=ylims[patient],axis_labels=True,title=patient)

p3 = viziums[2]

#%%% Apicome 
xlim = [970,1180]
ylim = [2840,2940]
fig, axes = plt.subplots(2,1,figsize=(6,6))

p3.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],title="Human colon")

p3["temp"] = p3["apicome"]
p3.adata.obs.loc[p3["nuc_cyto"]=="nuc","temp"] = "Nucleus"
p3.update_meta("temp",{"apical":"Apical","basal":"Basal"})
ax = p3.plot.spatial("temp",xlim=xlim,ylim=ylim,ax=axes[1],
                                    legend="lower left",legend_title="",
                                    cmap=["red","blue","gold"],alpha=0.3,exact=True)
ax.set_title(None)
plt.tight_layout()
plt.savefig(f"{path_output}/apicome_SPATIAL.pdf",bbox_inches='tight',pad_inches=0,dpi=300)
plt.savefig(f"{path_output}/apicome_SPATIAL.svg",bbox_inches='tight',pad_inches=0,dpi=300)
plt.savefig(f"{path_output}/apicome_SPATIAL.png",bbox_inches='tight',pad_inches=0,dpi=300)


#%%% Retention

fig, axes = plt.subplots(2,1,figsize=(7,7))

p3.plot.spatial(xlim=xlim,ylim=ylim,ax=axes[0],scalebar={"text_offset":0.06},title="Human colon")

p3["temp"] = p3["nuc_cyto"]
p3.update_meta("temp",{"nuc":"Nucleus","cyto":"Cytoplasm"})
ax = p3.plot.spatial("temp",xlim=xlim,ylim=ylim,scalebar={"text_offset":0.06},
                                   legend="lower left",legend_title="",
                                    cmap=["orange","cyan"],alpha=0.4,exact=True,ax=axes[1])
ax.set_title(None)
plt.tight_layout()

plt.savefig(f"{path_output}/retention_SPATIAL.pdf",bbox_inches='tight',pad_inches=0,dpi=300)
plt.savefig(f"{path_output}/retention_SPATIAL.svg",bbox_inches='tight',pad_inches=0,dpi=300)
plt.savefig(f"{path_output}/retention_SPATIAL.png",bbox_inches='tight',pad_inches=0,dpi=300)

#%% Correlation with LCM small intestines

human_lcm = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_colon\input\human_lcm.csv",index_col=0)
human_lcm = human_lcm[['expression_mean', 'log2_fc', 'qval']]
human_lcm.rename(columns={"log2_fc":"log2fc"},inplace=True)

human_lcm["gene"] = human_lcm.index
# combined_apicome["gene"] = combined_apicome.index
merged = (human_lcm.merge(combined_apicome,on='gene', suffixes=('_si', '_li')))

#%%% plot correlation
exp_thresh = 0
q_thresh = 0.25
# n_plot = 8
plot = merged.loc[(merged['expression_mean_si'] > exp_thresh) &
                   (merged['expression_mean_li'] > exp_thresh) &
                   (merged['qval_si'] <= q_thresh) &
                   (merged['qval_li'] <= q_thresh)].copy()

plot = plot.dropna()

# plot['dist_from_0_0'] = np.sqrt(plot['log2fc_si']**2 + plot['log2fc_li']**2)
# plot['quadrant'] = ((plot['log2fc_si'] > 0).astype(int).astype(str) + (plot['log2fc_li'] > 0).astype(int).astype(str)).map({'11': 'Q1', '01': 'Q2', '00': 'Q3', '10': 'Q4'})
# genes = plot.groupby('quadrant', group_keys=False).apply(lambda x: x.nlargest(n_plot, 'dist_from_0_0'))["gene"].tolist()
genes = ["PIGR","CDH1","DMBT1","CES2","KTN1","NET1","MYH14","KIF13B"]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_si", "log2fc_li" ,genes=genes,
                                  xlab="log2(apical/basal) - Small intestine",repel=True,title="mRNA localization in small and large intestines",
                                  ylab="log2(apical/basal) - Large intestine",color_genes="black",
                                  color="black",figsize=(6,6))
corr, pval = spearmanr(plot["log2fc_si"], plot["log2fc_li"])
pval = pval if pval > 0 else 1e-300

ax.text(0.05, 0.95,f"r = {corr:.2f}, p = {pval:.2g}",transform=ax.transAxes)


ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.pdf",bbox_inches='tight',pad_inches=0, dpi=300) 
ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.svg",bbox_inches='tight',pad_inches=0, dpi=300) 
ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.png",bbox_inches='tight',pad_inches=0, dpi=300) 

merged.to_csv(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.csv")


