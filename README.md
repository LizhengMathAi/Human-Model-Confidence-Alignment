# Human-Model-Confidence-Alignment

This repository contains the implementation code for the paper **"Human–Model Confidence Alignment: A Hidden Risk in Chest X-ray AI"**.

The paper identifies a reliability risk in deep learning models for medical imaging, where model confidence does not align with human-like behavior under image resolution degradation. It introduces the human–model confidence Alignment Rate (AR) metric and a preliminary fine-tuning solution to mitigate overconfident random-guessing behavior in chest X-ray classification and segmentation tasks, using the CheXpert dataset.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/LizhengMathAi/Human-Model-Confidence-Alignment.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Running `vit_chest_xray.py` for CheXpert experiments
- Running `CheXpert.ipynb` for Figure 2 data visualization
- Using `image-classification/run.sh` for ImageNet-1K experiments


For more details, refer to the paper.
