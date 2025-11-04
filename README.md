# VisionAssist - Real-Time Navigation Support for the Visually Impaired

## 🧭 Project Overview

   VisionAssist is a real-time navigation aid designed to help visually impaired individuals navigate their surroundings safely. Using object detection, tracking, and distance estimation, the system provides auditory feedback about obstacles and objects in the user’s path.
   The project uses a combination of **COCO 2017 dataset** and a **custom dataset** collected from **YouTube video frames** ([available here](https://drive.google.com/drive/folders/1ztLWfdN3As3kEFBYy0h9rb9OPw6CVTBp?usp=drive_link)) to train and fine-tune the YOLO model for real-world scenarios.
   The goal is to create an affordable and efficient assistive tool for enhanced independence and spatial awareness.

---

## 🧩 Problem Statement / Motivation

Visually impaired individuals often face challenges in perceiving their surroundings, especially in dynamic environments like streets or crowded areas. While existing assistive devices provide partial solutions, they are often expensive or lack real-time feedback. VisionAssist bridges this gap using computer vision and audio guidance to offer timely navigation support.

---

## 🧱 Repository Structure

```
Group-2-DS-and-AI-Lab-Project/
├── annotations/
│   ├── data.yaml
│   ├── instances_test.json
│   ├── instances_train.json
│   └── instances_val.json
├── docs/
│   ├── Milestone_1.pdf
│   ├── Milestone_2.pdf
│   ├── Milestone_3.pdf
│   └── Milestone_4.pdf
├── results/
│   └── eda/
│       ├── aspect_ratios.png
│       ├── bbox_areas.png
│       ├── class_distribution.png
│       ├── object_aspect_ratios.png
│       ├── object_locations_heatmap.png
│       └── objects_per_image.png
├── scripts/
│   ├── data_loading/
│   │   ├── .gitignore
│   │   ├── Custom Data Collection Script.ipynb
│   │   └── dataset_sample_collection_annotation.py
│   ├── training/
│   │   └── Main.ipynb
│   ├── DSAI_eval.ipynb
│   ├── EDA_MS_COCO.ipynb
│   └── Hyperparametertuning.ipynb
├── DATA_GOVERNANCE.md
└── README.md
```
---

## 🚀 Usage Guide

This section provides instructions on how to run and reproduce key parts of the VisionAssist project.

---

### 🧠 Model Training

Follow these steps to train the VisionAssist detection model using the [Main.ipynb](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/scripts/training/Main.ipynb) notebook.
These steps outline the workflow for dataset preparation, fine-tuning YOLOv8, and saving trained weights.

1. **Mount Google Drive in Colab**

   * Mount your Google Drive to access datasets.
   * Confirm that the shared **YouTube frames dataset** path exists ([Google Drive link](https://drive.google.com/drive/folders/1ztLWfdN3As3kEFBYy0h9rb9OPw6CVTBp?usp=drive_link)).
   * If not, adjust the directory path accordingly.

2. **Unzip COCO Data**

   * Extract `filtered_coco_data.zip` from Drive into a temporary folder.
   * Locate the extracted `images` and (if available) `labels` directories.

3. **Combine Datasets**

   * Create a folder named `master_dataset/images`.
   * Copy all COCO images into it.
   * Merge the **YouTube frame images** from the custom dataset.
   * Verify the total number of combined images.

4. **Auto-Annotate Using YOLOv8**

   * Install **Ultralytics** using:

     ```bash
     pip install ultralytics
     ```
   * Load the pretrained `yolov8n.pt` model.
   * Auto-annotate all images in `master_dataset/images` to generate YOLO TXT labels in `master_dataset/labels`.

5. **Split the Dataset**

   * Create `split_dataset/train`, `split_dataset/valid`, and `split_dataset/test` subfolders for both `images` and `labels`.
   * Shuffle and distribute files in a **70/20/10** ratio, ensuring each image has its corresponding label file.

6. **Generate Dataset Config YAML**

   * Create a `coco_custom_data.yaml` file that points to the split directories.
   * Include all 80 COCO classes in the `names:` section.

7. **Train YOLOv8 Model**

   * Run YOLOv8 training with desired hyperparameters:

     ```bash
     yolo task=detect mode=train model=yolov8n.pt data=coco_custom_data.yaml epochs=50 imgsz=640
     ```

8. **Save Trained Weights**

   * After training, copy the resulting `best.pt` file from `runs/detect/...` to your Google Drive for safekeeping.



---
📄 **Milestone Documents**

All official milestone submissions are located in the [`docs/`](./docs) directory of the repository:

* **[Milestone_1.pdf](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/docs/Milestone_1.pdf)** – Covers dataset selection, problem statement, and preliminary findings.
* **[Milestone_2.pdf](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/docs/Milestone_2.pdf)** – Details dataset preparation, preprocessing, exploration, and custom data collection.
* **[Milestone_3.pdf](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/docs/Milestone_3.pdf)** – Focuses on model selection, architecture choice, and training methodology.
* **[Milestone_4.pdf](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/docs/Milestone_4.pdf)** – Documents model training, hyperparameter tuning, evaluation, and application-level experimentation.
* **[Milestone_4 v2.pdf](https://github.com/22f2001061/Group-2-DS-and-AI-Lab-Project/blob/main/docs/Milestone_4 v2.pdf)** – Updates as per the feedback recieved on the M4 submission done in earlier iteration.


---
## 🧰 Technology Stack / Tools Used

   * **YOLOv8 (Ultralytics)** – Object detection
   * **ByteTrack** – Multi-object tracking
   * **gTTS (Google Text-to-Speech)** – Voice feedback
   * **OpenCV** – Image and video processing
   * **Python** – Core programming language
   * **MS COCO Dataset** – Base dataset for model training and benchmarking
   * **[Custom YouTube Frame Dataset](https://drive.google.com/drive/folders/1ztLWfdN3As3kEFBYy0h9rb9OPw6CVTBp?usp=drive_link)** – Additional dataset curated for fine-tuning and real-world diversity
   * **Jupyter Notebooks** – Development and experimentation environment

---


## 🧠 Model Description / Methodology

1. **Object Detection:** YOLOv8 detects objects in each video frame.
2. **Tracking:** ByteTrack assigns consistent IDs for moving objects.
3. **Distance Estimation:** Uses bounding box size and focal length for approximate object distance.
4. **Audio Feedback:** gTTS converts detections into spoken alerts for the user.

---

## 🧩 System Architecture / Workflow

1. Video Frame → YOLOv8 Detection → ByteTrack Tracking
2. Tracking Data → Distance Estimation → gTTS Audio Output
3. User receives real-time spoken navigation cues

---

## 👥 Team Members

* Balasurya K
* Jivraj Singh Shekhawat
* Tanuja Nair
* Karan Patil
* Prashasti Sarraf

---

## ⚖️ License

### a) Project License

This project is released under the **MIT License** — free for research and educational use.

### b) Dataset Credits and License

Annotations and dataset formats reference the **MS COCO dataset**, licensed under the **Creative Commons Attribution 4.0 License (CC BY 4.0)**.

---

## 🙏 Acknowledgements

We thank our instructors for their continuous guidance and the open-source community for providing powerful tools like Ultralytics YOLO and ByteTrack that made this project possible.
