# 🧩 Data Labeling and Visualization Tool

A lightweight web-based tool for viewing, labeling, and exploring tabular data interactively.  
Built with **Flask**, **Jinja2 templates**, and **vanilla CSS**, it provides an intuitive interface to inspect rows, toggle raw values, and select specific columns for display.

---

## 🚀 Features

- 📊 **Data Explorer:** Navigate and view rows or columns of a dataset.
- 🧭 **Side Navigation Panel:** Quickly jump between data rows or indexed column values.
- ⚙️ **Dynamic Column Selection:** Choose which columns to include or display.
- 🔍 **Raw Data Toggle:** Option to show or hide raw (unformatted) data values.
- 🎨 **Responsive UI:** Compact, scrollable layout with left sidebar and centered controls.

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, Jinja2, CSS
- **Data Handling:** Pandas (for reading and indexing datasets)
- **Environment:** Python 3.9+

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/nahidul-opu/Labeler.git
cd Labeler
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

Before running the app, you need to create a configuration file named **`.config`** in the root directory.  
This file defines application parameters, keyword classes, highlight color, and Gemini API credentials.

### Example `.config`
```ini
[APP]
KEYWORDS = ["K1", "K2", "K3"]
CLASSES = ["CLASS 1", "CLASS 2", "CLASS 3"]
HIGHLIGHT_COLOR = "#af7342ff"

[GEMINI]
MODEL = "gemini-2.0-flash"
API_KEY = "api-key"
PROMPT = ""
```

### Notes:
- **KEYWORDS**: list of key terms to highlight.  
- **CLASSES**: categories or labels used during annotation.  
- **HIGHLIGHT_COLOR**: defines the visual highlight color for detected keywords.  
- **API_KEY**: your Gemini API key from [Google AI Studio](https://aistudio.google.com/).  
- **PROMPT**: optional base instruction for model guidance (can be empty).  

> ⚠️ **Do not share or commit your API key.**  
> Add `.config` to `.gitignore`:
> ```bash
> echo ".config" >> .gitignore
> ```

---

## 🧩 Project Structure

```
project/
├── app.py
├── static/
│   └── style.css          # Unified CSS for all templates
├── templates/
│   ├── layout.html
│   ├── label.html
│   ├── select_columns.html
│   └── index.html
└── requirements.txt
```

---

## ⚙️ Running the App

1. Start the Flask server:

```bash
python app.py
```

2. Open in your browser:
```
http://127.0.0.1:5000/
```

---

## 🧭 Usage Guide

- **Step 1:** Upload or select a dataset.
- **Step 2:** Use **Select Columns** to choose which fields to display.
- **Step 3:** Use the left navigation bar to browse rows or a selected column.
- **Step 4:** Toggle the **“Show raw data”** checkbox to view raw values.
- **Step 5:** Label or review the dataset as needed.

---