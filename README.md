# 🛡️ WorkWithHelmetOnly

A computer vision application for detecting helmet usage on construction sites, built using YOLOv11, OpenCV, and Streamlit. The app provides real-time PPE compliance monitoring, automatic email alerts for violations, and a clean UI for reviewing results.

## 🚀 Features

- Real-time helmet detection using YOLOv11
- Multi-object tracking with SORT
- Violation alert via email (SMTP)
- Automatic compliance logging in CSV format
- Streamlit-based interactive dashboard
- Optimized for construction site monitoring

## 📘 User Manual

For a detailed step-by-step guide on using the application:  
👉 [Read the User Manual](https://user-manual-final.vercel.app/)

## 📂 Dataset

- **Source**: Roboflow PPE Dataset  
- **Images**: 1,613 (JPG/PNG format)  
- **Classes**: 8  
- **Split**: 70% Train / 20% Validation / 10% Test  

## 🧠 Technology Stack

- **YOLOv11** - Object Detection
- **OpenCV** - Image & video processing
- **SORT** - Object tracking
- **Python** - Core programming language
- **Streamlit** - Web app interface
- **SMTP** - Email notifications
- **CSV** - Compliance report output

## 🛠️ Installation

1. Clone the repository and navigate to the directory:

    ```bash
    cd WorkWithHelmetOnlyApp
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch with CUDA (for GPU acceleration):

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

4. Run the app:

    ```bash
    streamlit run webapp.py
    ```

## 📊 Performance Metrics

- Model tested using YOLOv11 on Roboflow dataset
- Accuracy metrics and confusion matrix per class available in the documentation
- Supports identification of:
  - True Positives & True Negatives
  - False Positives & False Negatives

## 📧 Contact

Created by **Abdullah Abo El-Hija** and **Zenab Abd Elghani**  
Advisor: Dr. Zeev Frenkel  
GitHub: [Final Project Repo](https://github.com/AbdullahAboElHija/FinalProject)

---

> **Note**: For best results, use a GPU-enabled machine for inference.
