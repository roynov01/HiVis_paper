# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 14:34:19 2025

@author: royno
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


from HiVis import HiVis

#%% Imports

path_organs = r"X:\roy\viziumHD\analysis\Python\version_11\organs"
# human_colon = pd.read_csv(rf"{path_organs}\human_colon\output_new\combined_pearson.csv",index_col=0)
# human_colon = human_colon[["qval","log2fc_mean","expression_mean_combined"]].copy()
# human_colon.rename(columns={"expression_mean_combined":"expression_mean","log2fc_mean":"log2fc"},inplace=True)

human_colon = pd.read_csv(rf"{path_organs}\human_colon\output_new\combined_apicome_epi_genes.csv",index_col=0)

human_colon["gene"] = human_colon.index

human_lcm = pd.read_csv(rf"{path_organs}\human_colon\input\human_lcm.csv",index_col=0)
human_lcm = human_lcm[['expression_mean', 'log2_fc', 'qval']]
human_lcm.rename(columns={"log2_fc":"log2fc"},inplace=True)

human_lcm["gene"] = human_lcm.index

merged = (human_lcm.merge(human_colon,on='gene', suffixes=('_si', '_li')))

#%% plot correlation
exp_thresh = 0
q_thresh = 0.25
n_plot = 8
plot = merged.loc[(merged['expression_mean_si'] > exp_thresh) &
                   (merged['expression_mean_li'] > exp_thresh) &
                   (merged['qval_si'] <= q_thresh) &
                   (merged['qval_li'] <= q_thresh)].copy()

plot = plot.dropna()

plot['dist_from_0_0'] = np.sqrt(plot['log2fc_si']**2 + plot['log2fc_li']**2)
plot['quadrant'] = ((plot['log2fc_si'] > 0).astype(int).astype(str) + (plot['log2fc_li'] > 0).astype(int).astype(str)).map({'11': 'Q1', '01': 'Q2', '00': 'Q3', '10': 'Q4'})
genes = plot.groupby('quadrant', group_keys=False).apply(lambda x: x.nlargest(n_plot, 'dist_from_0_0'))["gene"].tolist()

ax = HiVis.HiVis_plot.plot_scatter_signif(plot,"log2fc_si", "log2fc_li" ,genes=genes,
                                  xlab="log2(apical/basal) - Small intestine",repel=True,
                                  ylab="log2(apical/basal) - Large intestine",color_genes="black",
                                  color="black",x_line=0,y_line=0,figsize=(8,8))
corr, pval = spearmanr(plot["log2fc_si"], plot["log2fc_li"])

ax.set_title(f"r = {corr:.2f}, p = {pval:.2g}")

ax.get_figure().savefig(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.png", dpi=300, bbox_inches='tight') 
merged.to_csv(r"X:\roy\viziumHD\analysis\Python\version_11\comparison\output\apicome_intestines_cor.csv")











