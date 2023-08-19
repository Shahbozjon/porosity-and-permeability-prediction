# <p align="center"> AI for Porosity and Permeability Prediction from Geologic Core X-Ray Micro-Tomography </p>
### <p align="center"><em>Zangir Iklassov <sup>a</sup>, Dmitrii Medvedev <sup>a</sup>, Otabek Nazarov <sup>a</sup>, Shakhboz Razzokov <sup>b</sup></em></p>

<p><em><sup>a</sup> Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, United Arab Emirates</em></p>
<p><em><sup>b</sup> Gubkin Russian State University of Oil and Gas (National Research University) in Tashkent, Uzbekistan</em></p>

## 📌  Abstract
Geologic cores are rock samples that are extracted from deep under the ground during the well drilling process. They are used for petroleum reservoirs' performance characterization. Traditionally, physical studies of cores are carried out by the means of manual time-consuming experiments. With the development of deep learning, scientists actively started working on developing machine-learning-based approaches to identify physical properties without any manual experiments. Several previous works used machine learning to determine the porosity and permeability of the rocks, but either method was inaccurate or computationally expensive. We are proposing to use self-supervised pretraining of the very small CNN-transformer-based model to predict the physical properties of the rocks with high accuracy in a time-efficient manner. We show that this technique prevents overfitting even for extremely small datasets.

## 📌  Architecture
<p align="center"><img src="https://user-images.githubusercontent.com/69413364/206890626-5ebcb9ad-b41e-40cf-9b0f-84ea591ec50a.png" width="700" /> </p>
<p align="center"><b>Figure 2:</b> Models’ architectures (left - SSL model; right - supervised model)</p>

## 📌 Dataset
<em>11 Sandstones: raw, filtered and segmented data: </em> https://www.digitalrocksportal.org/projects/317
