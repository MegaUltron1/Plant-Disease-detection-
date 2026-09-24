# 🌿 Plant Disease Dataset

## 📌 Dataset Overview

This project uses the **PlantVillage Dataset**, sourced from Kaggle, for training, validating, and testing the plant disease classification model.

The dataset contains images of healthy and diseased plant leaves belonging to three crop categories: **Pepper Bell, Potato, and Tomato**. A total of **20,637 images** across **15 different classes** are used in this project.

The dataset is divided into training, validation, and testing sets to support model development and unbiased performance evaluation.

## 📊 Dataset Details

| Attribute | Details |
|---|---|
| Dataset Name | PlantVillage Dataset |
| Source | Kaggle |
| Total Images | 20,637 |
| Number of Classes | 15 |
| Crops Covered | Pepper Bell, Potato, Tomato |
| Training Split | 70% |
| Validation Split | 15% |
| Testing Split | 15% |
| Test Images | 3,109 |
| Model Input Size | 224 × 224 × 3 |
| Image Type | RGB Leaf Images |
| Task | Multi-Class Image Classification |

## 🌱 Dataset Classes

The dataset contains the following 15 healthy and diseased plant categories:

### 🫑 Pepper Bell
1. Pepper Bell — Bacterial Spot
2. Pepper Bell — Healthy

### 🥔 Potato
3. Potato — Early Blight
4. Potato — Late Blight
5. Potato — Healthy

### 🍅 Tomato
6. Tomato — Bacterial Spot
7. Tomato — Early Blight
8. Tomato — Late Blight
9. Tomato — Leaf Mold
10. Tomato — Septoria Leaf Spot
11. Tomato — Spider Mites (Two-Spotted Spider Mite)
12. Tomato — Target Spot
13. Tomato — Yellow Leaf Curl Virus
14. Tomato — Mosaic Virus
15. Tomato — Healthy

## 📂 Dataset Structure

The dataset is organized into three main directories: `train`, `valid`, and `test`.

Each directory contains separate subdirectories corresponding to the 15 classification classes.

```text
dataset/
│
├── train/
│   ├── Pepper__bell___Bacterial_spot/
│   ├── Pepper__bell___healthy/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   ├── Potato___healthy/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   ├── Tomato__Target_Spot/
│   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   ├── Tomato__Tomato_mosaic_virus/
│   └── Tomato_healthy/
│
├── valid/
│   └── [Same 15 class folders]
│
└── test/
    └── [Same 15 class folders]
