# Roads Polygon Detector App

A desktop application for AI-assisted road and lane polygon detection from imagery using deep learning.

The application combines semantic segmentation models with an intuitive graphical interface to detect road regions, generate polygon annotations, and export results in XML format for downstream GIS or annotation workflows.

---

## Overview

Roads Polygon Detector App was developed to simplify the process of identifying road surfaces and lane markings from images.

Instead of manually outlining roads, users can load an image, run AI inference, review the generated polygons, and export the results for further processing.

The application provides an easy-to-use desktop interface while leveraging PyTorch-based segmentation models behind the scenes.

---

## Key Features

- Desktop GUI built with PyQt
- AI-powered road segmentation
- Lane detection support
- Automatic polygon generation
- XML annotation export
- Interactive visualization of predictions
- Lightweight desktop workflow
- Supports custom trained model weights

---

## Application Workflow

```text
Input Image
      │
      ▼
Load into Application
      │
      ▼
Run Deep Learning Model
      │
      ▼
Road & Lane Segmentation
      │
      ▼
Polygon Generation
      │
      ▼
Visualization
      │
      ▼
XML Export
```

---

## Project Structure

```text
Roads-Polygon-Detector-App/
│
├── gui.py                  # Desktop user interface
├── model.py                # Deep learning inference
├── utils.py                # Utility functions
├── ui/                     # Qt Designer UI files
├── models/                 # Pretrained model weights
├── Images/                 # Sample images
├── icons/                  # Application assets
└── README.md
```

---

## Technologies

- Python
- PyTorch
- PyQt5
- OpenCV
- NumPy
- Pillow
- XML Processing

---

## Installation

Clone the repository

```bash
git clone https://github.com/bilalahhmedd/raods-polygon-detector-app.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python gui.py
```

---

## Usage

1. Launch the application.
2. Load an input image.
3. Execute road detection.
4. Review generated polygons.
5. Export annotation results as XML.

---

## Output

The application generates:

- Road segmentation results
- Lane detection visualization
- Polygon boundaries
- XML annotation files

---

## Typical Use Cases

- Road extraction
- Infrastructure mapping
- Dataset preparation
- AI annotation workflows
- Computer vision research
- GIS preprocessing

---

## Future Improvements

Some planned improvements include:

- GeoJSON export
- Batch image processing
- REST API
- Model management
- Configuration system
- Additional export formats
- Improved polygon editing tools

---

## License

This repository contains code developed for a client project and is shared for demonstration purposes. Please contact the repository owner regarding commercial usage or redistribution.