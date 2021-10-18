## crc-lnm
**Deep learning can predict lymph node status directly from histology in colorectal cancer**


![Pipeline](/images/pipeline.jpg)


## **Abstract**
**Background:** Lymph node status is a prognostic marker and strongly influences therapeutic decisions in colorectal cancer (CRC).  
**Objectives:** To investigate whether image features extracted by a deep learning model from routine histological slides and/or clinical data can be used to predict CRC lymph node metastasis (LNM).  
**Methods:** Using histological whole slide images (WSIs) of primary tumors of 2431 patients in the DACHS cohort, we trained a convolutional neural network (CNN) to predict LNM. In parallel, we used clinical data derived from the same cases in logistic regression analyses. Subsequently, the slide-based artificial intelligence predictor (SBAIP) score was included in the regression. WSIs and data from 582 patients of the TCGA cohort were used as external test set.  
**Results:** On the internal test set, the SBAIP achieved an area under receiver operating characteristic (AUROC) of 71.0%, the clinical classifier achieved an AUROC of 67.0% and a combination of the two classifiers yielded an improvement to 74.1%. Whereas the clinical classifier’s performance remained stable on the TCGA set, performance of the SBAIP dropped to an AUROC of 61.2%. Performance of the clinical classifier depended strongly on the T stage.  
**Conclusion:** Deep learning-based image analysis may help predict LNM of CRC patients using routine histological slides. Combination with clinical data such as T stage might be useful. Strategies to increase performance of the SBAIP on external images should be investigated.  


## **Disclaimer**  
Everything in this repository is provided without any guarantees. We share our code to establish transparency. 
You can use it in any way you like. We will not maintain this repository. Please do not submit any issues or pull requests.


## Citations

```bibtex
@article{KIEHL2021464,
title = {Deep learning can predict lymph node status directly from histology in colorectal cancer},
journal = {European Journal of Cancer},
volume = {157},
pages = {464-473},
year = {2021},
issn = {0959-8049},
doi = {https://doi.org/10.1016/j.ejca.2021.08.039},
url = {https://www.sciencedirect.com/science/article/pii/S0959804921005700},
author = {Lennard Kiehl and Sara Kuntz and Julia Höhn and Tanja Jutzi and Eva Krieghoff-Henning and Jakob N. Kather and Tim Holland-Letz and Annette Kopp-Schneider and Jenny Chang-Claude and Alexander Brobeil and Christof {von Kalle} and Stefan Fröhling and Elizabeth Alwers and Hermann Brenner and Michael Hoffmeister and Titus J. Brinker},
keywords = {Colorectal cancer, Lymph node status, Deep learning, CNN, Clinical data}
}
```
