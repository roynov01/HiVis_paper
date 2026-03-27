# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:23:35 2026

@author: royno
"""


from HiVis import HiVis

import anndata as ad

path_out = ""
path = ""

ad1 = ad.read_h5ad(f"{path}/liver_rep1_HiVis.h5ad")
ad2 = ad.read_h5ad(f"{path}/liver_rep2_HiVis.h5ad")

img_fullres1 = f"{path}/liver_rep1_fullres.tif"
img_highres1 = f"{path}/liver_rep1_highres.tif"
img_lowres1 = f"{path}/liver_rep1_lowres.tif"
img_fullres2 = f"{path}/liver_rep2_fullres.tif"
img_highres2 = f"{path}/liver_rep2_highres.tif"
img_lowres2 = f"{path}/liver_rep2_lowres.tif"

json1 = f"{path}/liver_rep1_scalefactors_json.json"
json2 = f"{path}/liver_rep2_scalefactors_json.json"

name1 = "Mouse1"
name2 = "Mouse2"

properties1 = {"organism":"mouse","organ":"liver"}
properties2 = {"organism":"mouse","organ":"liver"}


mouse1 = HiVis.HiVis(ad1, img_fullres1, img_highres1, img_lowres1, json1, name1, path_out, properties1,plot_qc=False)
mouse2 = HiVis.HiVis(ad2, img_fullres2, img_highres2, img_lowres2, json2, name2, path_out, properties2,plot_qc=False)







    