# Neural Network Optimization from Scratch (NumPy Implementation) 🚀

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Core_Logic-013243?style=for-the-badge&logo=numpy)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

This project is a comprehensive implementation of a Multi-Layer Perceptron (MLP) and various optimization algorithms **built entirely from scratch using NumPy**, without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

The project covers the entire pipeline: from synthetic data generation using a local LLM (Gemma-9B), to semantic vectorization (Embeddings), manual backpropagation calculus, and 2D visualization of optimization trajectories using t-SNE.

## 🎯 Project Goal

* **Mathematical Depth:** Understanding the core mathematical operations of AI (derivatives, chain rule, matrix multiplications) by coding them manually instead of using `model.fit()`.
* **Optimization Benchmark:** Comparing the performance, convergence speed, and stability of **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, **Adam**, **AdaGrad**, and **RMSProp**.
* **Visualization:** Analyzing how algorithms navigate the "Loss Landscape" by reducing high-dimensional weight updates to 2D using **t-SNE**.

---

## 🛠️ Technologies & Methodology

All components below are implemented "from scratch" using Python and NumPy.

### 1. Synthetic Data Generation
* **Model:** Local **Gemma-9B** model running via **Ollama**.
* **Task:** Generated a regression-based classification dataset consisting of Question-Answer pairs.
* **Dataset:** Labeled as incorrect answers (-1) and correct answers (+1).

### 2. Semantic Embeddings
* Compared word-based (TF-IDF) vs. semantic (BERT/Transformer) approaches.
* **Model:** Used `ytu-ce-cosmos/turkish-e5-large` to convert text into a 1024-dimensional vector space.
* **Preprocessing:** Concatenated Question + Answer vectors and added a bias term, resulting in a **2049-dimensional input**.

### 3. Model Architecture (NumPy Only)
* **TwoLayerMLP:**
    * **Input Layer:** 2049 neurons.
    * **Hidden Layer:** 64 neurons (Hyperparameter tuned) with **Tanh** activation.
    * **Output Layer:** 1 neuron with **Tanh** activation.
* **RecursiveMLP:** Implemented a recursive structure to support dynamic depth and arbitrary layer configurations.

---

## 📊 Algorithm Benchmarks

Based on 100 Epochs of training, here are the characteristics of the implemented optimizers:

| Optimizer | Convergence Speed | Stability | Test Accuracy | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **GD** | 🔴 Slow | 🟢 Very High | 🟡 Low (~0.62) | Processes the entire dataset at once. Smooth path but very slow convergence. |
| **SGD** | 🟡 Medium | 🔴 Low | 🟢 Good (~0.80) | Uses mini-batches (32). Noisy trajectory (zig-zag) due to stochastic nature. |
| **Adam** | 🟢 **Very Fast** | 🟢 **High** | 🌟 **Excellent (~0.98)** | Uses Momentum and Adaptive Learning Rates. Reaches the global minimum quickly and stably. |

---

## 📈 Results & Visualizations

### 1. Optimization Trajectories (t-SNE Analysis)
The visualization below shows how different algorithms navigate the high-dimensional loss landscape (reduced to 2D).
* **Adam:** Takes the most direct and stable path to the target.
* **SGD:** Oscillates around the target due to noise.
* **GD:** Moves slowly and linearly.

*(Add your `tsne_trajectory.png` here from the report)*

### 2. Loss & Accuracy Curves
Training loss decay and test accuracy improvements over epochs:

*(Add your `loss_graph.png` here from the report)*

---

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zeynelbyhn/ML-OptimizationTechniques.git](https://github.com/zeynelbyhn/ML-OptimizationTechniques.git)
    cd ML-OptimizationTechniques
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn sentence-transformers
    ```

3.  **Run the training and benchmark:**
    ```bash
    python main.py
    ```

---

## 🧠 Theory: How It Works?

The model implements **Backpropagation** via manual calculus and chain rule derivatives.

**General Weight Update Rule:**
$$W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}$$

**Adam Optimization Formula (As Implemented):**
Adam combines the benefits of Momentum (moving average of gradients) and RMSProp (moving average of squared gradients):
1.  $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (Momentum)
2.  $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (Velocity)
3.  $W = W - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*Developed at YTU CE (Computer Engineering).*
