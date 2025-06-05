# Official PyTorch implementation of LSMI_Estimator

This repository provides the official PyTorch implementation for the paper **"Efficient Quantification of Multimodal Interaction at Sample Level"** (ICML 2025).
Our work introduces the Lightweight Sample-wise Multimodal Interaction (LSMI) Estimator, a method to efficiently quantify and distinguish redundancy, uniqueness, and synergy at the sample level in multimodal data.

**Paper Title:** "Efficient Quantification of Multimodal Interaction at Sample Level"

**Authors:** [Zequn Yang](https://bjlfzs.github.io/), [Hongfa Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=q9Fn50QAAAAJ), [Di Hu](https://dtaoo.github.io/index.html)

**Accepted by:** Forty-Second International Conference on Machine Learning (ICML 2025)

## Method

LSMI aims to decompose the task-relevant information from two modalities, $x_1$ and $x_2$, with respect to a target $y$, into three distinct components:
*   **Redundancy ($r$)**: Information about $y$ shared between $x_1$ and $x_2$.
*   **Uniqueness ($u_1, u_2$)**: Information about $y$ unique to $x_1$ (or $x_2$).
*   **Synergy ($s$)**: Information about $y$ that emerges only when $x_1$ and $x_2$ are considered jointly.

These pointwise interactions are related by the following equations:
$$i(x_1; y) = r + u_1,$$
$$i(x_2; y) = r + u_2,$$
$$i(x_1, x_2; y) = r + u_1 + u_2 + s,$$
where $i(x;y)$ denotes the pointwise mutual information.

<div align="center">
  <img src="figure/figure_1.jpg" width="50%">
  <p>Figure 1: Illustration of sample-level multimodal interactions, depicting redundancy ($r$), uniqueness ($u_1, u_2$), and synergy ($s$) as components of the total multimodal information $i(x_1, x_2; y)$.</p>
</div>

The unique determination of these interactions hinges on a pointwise definition of redundancy ($r$). Redundancy is derived using an information decomposition framework, as illustrated in Figure 2. This framework traces information flow through a lattice structure to identify redundant components, ensuring that information quantities monotonically decrease along the decomposition path.

<div align="center">
  <img src="./figure/figure_2.jpg" alt="Figure 2: Redundancy Estimation Framework" width="80%">
  <p>Figure 2: The Redundancy Estimation Framework. Information flow is traced through a lattice structure to identify redundant components, ensuring monotonic decrease of information quantities along the decomposition path.</p>
</div>

Our approach estimates redundancy by leveraging information flow, ensuring monotonicity. Specifically, pointwise mutual information is decomposed into $i^+$ and $i^-$ for each modality, satisfying the framework; redundancy is then determined for each part and combined to yield the overall redundancy interaction.

For continuous distributions, interactions are quantified using KNIFE (Pichler et al., 2022) for efficient differential entropy estimation. This provides $h_{\theta}(x)$ as an estimate for $h(x)$ (the $i^+$ component) and facilitates the estimation of the $i^-$ component. This lightweight methodology is well-suited for sample-level analysis.

## Getting Started

### Requirements
- Python 3.8
<pre><code>
pip install -r requirements.txt
</code></pre>

### Running the Demo
To run the LSMI_Estimator demo:
<pre><code>
python main_lsmi.py
</code></pre>
The `main_lsmi.py` script is the primary entry point for experiments. Algorithm parameters and dataset configurations can be modified within this script.

### Data Preparation
Data for the LSMI Estimator must be provided as a PyTorch tensor file (`.pt`). The `get_loader` function in `utils.py` handles data loading from this file. The file should contain a dictionary with the following keys for training and validation sets:

-   `'train_modal_1_features'`: Features for the first modality (training set).
-   `'train_modal_2_features'`: Features for the second modality (training set).
-   `'train_targets'`: Target labels (training set).
-   `'val_modal_1_features'`: Features for the first modality (validation set).
-   `'val_modal_2_features'`: Features for the second modality (validation set).
-   `'val_targets'`: Target labels (validation set).

An example script, `gaussian_data.py`, demonstrates the generation of synthetic data from a mixed Gaussian distribution.

For custom or complex datasets:
1.  Extract features (e.g., using pre-trained models) to obtain unimodal and multimodal representations.
2.  Save these features in the specified PyTorch tensor file format (`.pt`) with the keys listed above.
3.  Adapt the data loading process by modifying the `data_generate` function in `main_lsmi.py` as necessary.

## Citation

If you find this work useful in your research, please consider citing our paper:
<pre><code>
@inproceedings{yang2025Efficient,
  title={Efficient Quantification of Multimodal Interaction at Sample Level},
  author={Yang, Zequn and Wang, Hongfa and Hu, Di},
  booktitle={Forty-Second International Conference on Machine Learning},
  year={2025}
}
</code></pre>

## Acknowledgement

This work is sponsored by the CCF-Tencent Rhino-Bird Open Research Fund, the National Natural Science Foundation of China (Grant No. 62106272), the Public Computing Cloud of Renmin University of China, and the fund for building world-class universities (disciplines) of Renmin University of China.

## Contact

If you have any detailed questions or suggestions, you can email us:
**zqyang@ruc.edu.cn**