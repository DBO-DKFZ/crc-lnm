## crc-lnm
**Prediction of lymph node status from routine histological slides of colorectal cancer using deep learning**


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
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```