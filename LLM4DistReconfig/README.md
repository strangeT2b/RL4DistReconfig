# LLM4DistReconfig

## A Fine-tuned Large Language Model for Power Distribution Network Reconfiguration

🚀 **Accepted at NAACL 2025 Main Track**  
🔗 **Paper:** [LLM4DistReconfig](https://arxiv.org/abs/2501.14960)  
📡 **Developed with Resources from:** [NYU HPC](https://sites.google.com/nyu.edu/nyu-hpc/)  

---

## 🔥 Introduction
**LLM4DistReconfig** is a fine-tuned **Llama 3.1** model designed to solve **grid reconfiguration** tasks for power distribution systems. Our model has been rigorously tested on various network sizes, both individually and combined, and has been evaluated on **unseen network datasets**, including sizes that were both **within** and **outside** of the training distribution.

We provide a **robust and automated** framework that enables **seamless fine-tuning and evaluation** of the model while allowing for easy modifications to adapt to different requirements. This repository contains:

- **Pre-configured Python notebooks** for dataset generation, model training, and evaluation.
- **Automated scripts** for preparing datasets and fine-tuning models.
- **Customizable loss functions** to improve model performance and adaptability.

---

## 📌 Requirements
To run the code, install the required dependencies:
```bash
pip install torch accelerate bitsandbytes peft transformers trl
```

---

## 📂 Datasets
### 🔗 **Accessing Data**
The dataset files can be found at [grid-datasets](https://github.com/panaschristou/grid-datasets). Download and place them inside a folder named `csv_files` within `Dataset-Notebooks`. If the folder does not exist, create it and add the CSV files.

### 📜 **Dataset Preparation**
- Inside `Dataset-Notebooks`, you'll find the **dataset-generation-script** notebook, which processes each dataset by:
  - Creating the **prompt**, **input**, and **output** for the model.
  - Splitting data into **training, validation, and testing** sets.
  - Generating **auxiliary files** required for training.
  - Combining different dataset sizes into a **single unified dataset**.

### 🔄 **MATLAB Data Conversion**
Since our datasets were generated using **MATPOWER** in MATLAB, we provide a **MATLAB-to-Python conversion script** to easily transform `.mat` files into the required format for dataset generation. This ensures a **smooth workflow** for integrating custom datasets.

---

## 🎯 Fine-tuning
### 📌 **Setup**
Navigate to the `Model-Notebooks` folder, where you will find templates for fine-tuning Llama 3.1 on your dataset. Before starting, ensure that:
- Your dataset files are **fully prepared**.
- The `.sh` script is modified with the correct **dataset path** and **model path**.
- You specify where to **save the fine-tuned model** and configure relevant **hyperparameters**.

### 🔧 **Customizable Hyperparameters**
During fine-tuning, you can adjust:
- **Learning Rate**
- **Loss Function Components:**
  - **Invalid Edges Loss**: Penalizes invalid edges in the output.
  - **Subgraphs Loss**: Penalizes disconnected subgraphs.
  - **Cycles Loss**: Penalizes cycles in the reconfigured grid.
- You can use **any combination** of these losses to tailor the training process.

---

## 📊 Model Evaluation
### 🚀 **Evaluation Workflow**
We provide templates in `Model-Notebooks` to evaluate your trained model. You can:
- Load the **fine-tuned model** or a baseline model from **Hugging Face** for comparison.
- Specify the **number of samples** for evaluation.
- Save results in the `evaluations` folder as:
  - **Text responses** (model outputs)
  - **Metrics in CSV format**

### 📈 **Precomputed Baselines**
For easier comparison, we provide scripts to evaluate standard models such as **Falcon, Mistral, and Llama** against your fine-tuned version.

### ⚡ **Alternative Evaluation Approach**
Inside `Dataset-Notebooks`, we include a step-by-step evaluation script that allows you to compute metrics **without queueing a job**, providing a more interactive debugging experience.

---

## 🎯 Why Use LLM4DistReconfig?
✔ **Fully Automated Pipeline** – From dataset processing to model evaluation.  
✔ **Highly Customizable** – Modify loss functions, hyperparameters, and datasets with ease.  
✔ **Supports Multiple Architectures** – Compare results with various transformer models.  
✔ **Optimized for Power Grids** – Specifically designed for distribution network reconfiguration.  

---

## 🏆 Acknowledgments
We extend our gratitude to **NYU HPC** for their continuous support in resolving issues and allocating GPU resources that made this research possible.

---

## 🚀 Get Started
Clone the repository and start training your own LLM for power grid reconfiguration!
```bash
git clone https://github.com/panaschristou/LLM4DistReconfig.git
cd LLM4DistReconfig
```

Let’s **reconfigure the grid with AI!** ⚡🤖
