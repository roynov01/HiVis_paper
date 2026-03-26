# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:17:13 2025

@author: royno
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#%% Imports
cols = ["qval","log2fc","expression"]
path_organs = r"X:\roy\viziumHD\analysis\Python\version_11\organs"
human_liver = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_liver\output\human_liver_M6_DGE_retention_midzones.csv",index_col=0)
human_liver.rename(columns={"expression_mean":"expression"},inplace=True)
human_liver = human_liver[cols+["gene"]]
# mouse_liver = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_subset_Retention_allzones_combinedBothMice_epi_genes.csv",index_col=0)
mouse_liver = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_retention_allzones.csv",index_col=0)


# mouse_liver = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\mouse_liver\output\WT\mouse_liver_98_WT_retention.csv",index_col=0)
# mouse_liver.rename(columns={"log2fc_allzones":"log2fc","expression_min_allzones":"expression","qval_allzones":"qval"},inplace=True)
mouse_liver.rename(columns={"expression_min":"expression"},inplace=True)

mouse_liver = mouse_liver[cols]

mouse_liver["gene"] = mouse_liver.index
mouse_liver.index.name=None
human_colon = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\organs\human_colon\output_new\combined_retention_epi_genes.csv",index_col=0)
human_colon.rename(columns={"expression_mean":"expression"},inplace=True)
human_colon = human_colon[cols]

human_colon["gene"] = human_colon.index
human_colon.index.name=None

#%% Liver - human mouse comparison
#%%% Orthology based human-mouse conversion
ensamble = pd.read_csv(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\Mouse_Human_orthology_ENS109.csv")

orthology_ok = ensamble[ensamble['Human orthology confidence [0 low, 1 high]'] == 1]

mouse2human = (
    orthology_ok
    .dropna(subset=['Human gene name'])
    .set_index('Gene name')['Human gene name']
    .to_dict()
)

mouse_conv = mouse_liver.copy()
mouse_conv['gene_mouse'] = mouse_conv['gene']
mouse_conv['gene'] = mouse_conv['gene'].map(mouse2human)
del mouse_conv['gene_mouse']

# Discard rows where no orthologue was found
mouse_conv = mouse_conv.dropna(subset=['gene'])

# Merge
merged = (human_liver.merge(mouse_conv,on='gene', suffixes=('_human', '_mouse')))


#%%% Plot correlation

exp_thresh = 1e-4
qval_thresh = 0.05
n_plot = 30
plot = merged.loc[(merged['expression_human'] > exp_thresh) &(merged['qval_human'] <= qval_thresh) &
                   (merged['expression_mouse'] > exp_thresh) & (merged['qval_mouse'] <= qval_thresh)].copy()

plot = plot.dropna()

genes = ["FTH1","NDUFA1","MLXIPL","VEGFA","CYP3A43","ABCC2","FTCD","PID1","BHLHE40","RGN","BDH1","GSTK1","FADS2","SYVN1"]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_mouse", "log2fc_human" ,genes=genes,
                                  xlab="log2(nucleus/cytoplasm) - Mouse",repel=True,title="Nuclear retention across species - liver",
                                  ylab="log2(nucleus/cytoplasm) - Human",color_genes="black",
                                  color="black",x_line=0,y_line=0,figsize=(6,6))
corr, pval = spearmanr(plot["log2fc_mouse"], plot["log2fc_human"])
pval = pval if pval > 0 else 1e-300

ax.text(0.55, 0.04,f"r = {corr:.2f}, p = {pval:.2g}",transform=ax.transAxes)

ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\retention_liver_cor.pdf", 
                        bbox_inches='tight',pad_inches=0) 
merged.to_csv(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\retention_liver_cor.csv")


#%% Human liver/colon comparison

colon_liver = (human_liver[["qval","log2fc","expression","gene"]].merge(human_colon,on='gene', suffixes=('_liver', '_colon')))


exp_thresh = 1e-4
qval_thresh = 0.05
# n_plot = 12
plot = colon_liver.loc[(colon_liver['expression_liver'] > exp_thresh) & (colon_liver['qval_liver'] < qval_thresh) &
                   (colon_liver['expression_colon'] > exp_thresh)& (colon_liver['qval_colon'] < qval_thresh)].copy()

plot = plot.dropna()


genes=["SLC40A1","FTH1","FTL","KTN1","FABP1","LENG8","CYP3A5","VEGFA","ABCA5","LPP","SLC25A13","CRYL1","SPAG9","SLC16A1"]
ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_liver", "log2fc_colon" ,
                                          genes=genes,
                                  xlab="log2(nucleus/cytoplasm) - Liver",repel=True,title="Nuclear retention in liver and colon",
                                  ylab="log2(nucleus/cytoplasm) - Colon",color_genes="black",
                                  color="black",x_line=0,y_line=0,figsize=(6,6))
corr, pval = spearmanr(plot["log2fc_liver"], plot["log2fc_colon"])
pval = pval if pval > 0 else 1e-300

ax.text(0.5, 0.04,f"r = {corr:.2f}, p = {pval:.2g}",transform=ax.transAxes)

ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\retention_colon_liver.pdf", 
                        bbox_inches='tight',pad_inches=0) 
merged.to_csv(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\retention_colon_liver.csv")


