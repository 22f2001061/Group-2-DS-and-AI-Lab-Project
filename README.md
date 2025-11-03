# VisionAssist - Real-Time Navigation Support for the Visually Impaired

## 🧭 Project Overview

VisionAssist is a real-time navigation aid designed to help visually impaired individuals navigate their surroundings safely. Using object detection, tracking, and distance estimation, the system provides auditory feedback about obstacles and objects in the user’s path. The goal is to create an affordable and efficient assistive tool for enhanced independence and spatial awareness.

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

## 🧰 Technology Stack / Tools Used

* **YOLOv8 (Ultralytics)** – Object detection
* **ByteTrack** – Multi-object tracking
* **gTTS (Google Text-to-Speech)** – Voice feedback
* **OpenCV** – Image and video processing
* **Python** – Core programming language
* **MS COCO Dataset** – Pre-trained annotation structure reference
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
