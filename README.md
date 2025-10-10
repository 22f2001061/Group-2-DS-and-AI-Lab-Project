
---

## Project Overview

### [`/docs`](./docs)
Contains documentation and milestone submissions.  
- 📄 **[Milestone_1.pdf](./docs/Milestone_1.pdf)** – Official submission report for Milestone 1, covering dataset selection, problem statement, and preliminary findings.

- 📄 **[Milestone_2.pdf](./docs/Milestone_2.pdf)** – Official submission report for Milestone 2, covering dataset preparation, preprocessing, Exploration and custom data collection.

---

### [`/images`](./images)
This directory is used to store supporting figures or visual references generated during dataset preparation or modeling.  
*(Currently empty; future visual outputs such as model architectures or result snapshots can be placed here.)*

---

### [`/results`](./results)
Houses analysis outputs, evaluation graphs, and result visualizations.

#### [`/results/eda`](./results/eda)
Contains all **Exploratory Data Analysis (EDA)** visualizations generated from the COCO dataset:
- 🧮 [aspect_ratios.png](./results/eda/aspect_ratios.png): Distribution of object aspect ratios  
- 📦 [bbox_areas.png](./results/eda/bbox_areas.png): Bounding box area distribution  
- 🏷️ [class_distribution.png](./results/eda/class_distribution.png): Frequency of object categories  
- 🖼️ [objects_per_image.png](./results/eda/objects_per_image.png): Count of objects per image  
- 🧱 [object_aspect_ratios.png](./results/eda/object_aspect_ratios.png): Ratio variation across categories  
- 🌍 [object_locations_heatmap.png](./results/eda/object_locations_heatmap.png): Spatial density of objects in the dataset

These plots were generated using the `EDA_MS_COCO.ipynb` notebook.

---

### [`/scripts`](./scripts)
Contains all core Jupyter notebooks used in data collection, processing, and analysis.

- 📘 **[Custom Data Collection Script.ipynb](./scripts/Custom%20Data%20Collection%20Script.ipynb)**  
  Demonstrates how to collect or extend dataset samples with custom annotations or additional sources.

- 📗 **[dataset_sample_collection_annotations_and_examples.ipynb](./scripts/dataset_sample_collection_annotations_and_examples.ipynb)**  
  Provides examples of dataset annotation formats, data sample visualization, and metadata structure from COCO.

- 📙 **[EDA_MS_COCO.ipynb](./scripts/EDA_MS_COCO.ipynb)**  
  Performs a detailed **Exploratory Data Analysis** on COCO annotations, including:
  - Object count and class distribution  
  - Bounding box statistics  
  - Heatmaps and aspect ratio visualizations  
  - Insights for data quality and balance

---

## 📜 Supporting Files

- **[DATA_GOVERNANCE.md](./DATA_GOVERNANCE.md)** – Outlines dataset licensing, ethical standards, and compliance with IITM MLOps project guidelines.  
- *(Optional)* **data.yaml** – To be added for dataset metadata (source, license, categories, and paths).

---

## 🧠 Project Summary

This project involves analyzing and curating the **COCO 2017 dataset** for computer vision model development.  
Key objectives include:
- Performing detailed **EDA** on dataset annotations.  
- Ensuring **data ethics and governance compliance**.  
- Preparing clean, well-documented assets for downstream MLOps pipelines.

---

## 🚀 Next Steps

- Incorporate `data.yaml` for dataset metadata tracking.  
- Integrate pre-processing and model training pipelines for Milestone 3.  
- Expand documentation with experiment logs and model evaluation metrics.

---

