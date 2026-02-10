# Neural Network Optimization from Scratch (NumPy Implementation) 🚀

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Core_Logic-013243?style=for-the-badge&logo=numpy)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

[cite_start]This project is a comprehensive implementation of a Multi-Layer Perceptron (MLP) and various optimization algorithms **built entirely from scratch using NumPy**, without relying on high-level deep learning frameworks like PyTorch or TensorFlow[cite: 131, 327].

[cite_start]The project covers the entire pipeline: from synthetic data generation using a local LLM (Gemma-9B), to semantic vectorization (Embeddings), manual backpropagation calculus, and 2D visualization of optimization trajectories using t-SNE[cite: 2, 114, 1059].

## 🎯 Project Goal

* [cite_start]**Mathematical Depth:** Understanding the core mathematical operations of AI (derivatives, chain rule, matrix multiplications) by coding them manually instead of using `model.fit()`[cite: 197, 198].
* [cite_start]**Optimization Benchmark:** Comparing the performance, convergence speed, and stability of **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**, **Adam**, **AdaGrad**, and **RMSProp**[cite: 327, 714, 720].
* [cite_start]**Visualization:** Analyzing how algorithms navigate the "Loss Landscape" by reducing high-dimensional weight updates to 2D using **t-SNE**[cite: 1061].

---

## 🛠️ Technologies & Methodology

All components below are implemented "from scratch" using Python and NumPy.

### 1. Synthetic Data Generation
* [cite_start]**Model:** Local **Gemma-9B** model running via **Ollama**[cite: 2, 3].
* [cite_start]**Task:** Generated a regression-based classification dataset consisting of Question-Answer pairs[cite: 18, 19].
* [cite_start]**Dataset:** Labeled as incorrect answers (-1) and correct answers (+1)[cite: 29].

### 2. Semantic Embeddings
* [cite_start]Compared word-based (TF-IDF) vs. semantic (BERT/Transformer) approaches[cite: 742].
* [cite_start]**Model:** Used `ytu-ce-cosmos/turkish-e5-large` to convert text into a 1024-dimensional vector space[cite: 114].
* [cite_start]**Preprocessing:** Concatenated Question + Answer vectors and added a bias term, resulting in a **2049-dimensional input**[cite: 116, 121].

### 3. Model Architecture (NumPy Only)
* **TwoLayerMLP:**
    * [cite_start]**Input Layer:** 2049 neurons[cite: 132].
    * [cite_start]**Hidden Layer:** 64 neurons (Hyperparameter tuned) with **Tanh** activation[cite: 133, 151].
    * [cite_start]**Output Layer:** 1 neuron with **Tanh** activation[cite: 136, 153].
* [cite_start]**RecursiveMLP:** Implemented a recursive structure to support dynamic depth and arbitrary layer configurations[cite: 881].

---

## 📊 Algorithm Benchmarks

[cite_start]Based on 100 Epochs of training, here are the characteristics of the implemented optimizers[cite: 328, 538]:

| Optimizer | Convergence Speed | Stability | Test Accuracy | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **GD** | 🔴 Slow | 🟢 Very High | 🟡 Low (~0.62) | Processes the entire dataset at once. [cite_start]Smooth path but very slow convergence[cite: 380]. |
| **SGD** | 🟡 Medium | 🔴 Low | 🟢 Good (~0.80) | Uses mini-batches (32). [cite_start]Noisy trajectory (zig-zag) due to stochastic nature[cite: 357, 493]. |
| **Adam** | 🟢 **Very Fast** | 🟢 **High** | 🌟 **Excellent (~0.98)** | Uses Momentum and Adaptive Learning Rates. [cite_start]Reaches the global minimum quickly and stably[cite: 494, 540]. |

---

## 📈 Results & Visualizations

### 1. Optimization Trajectories (t-SNE Analysis)
The visualization below shows how different algorithms navigate the high-dimensional loss landscape (reduced to 2D).
* **Adam:** Takes the most direct and stable path to the target.
* **SGD:** Oscillates around the target due to noise.
* **GD:** Moves slowly and linearly.

[cite_start]*(Add your `tsne_trajectory.png` here from the report)* [cite: 1063, 1065, 1067]

### 2. Loss & Accuracy Curves
Training loss decay and test accuracy improvements over epochs:

[cite_start]*(Add your `loss_graph.png` here from the report)* [cite: 383, 406]

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

[cite_start]The model implements **Backpropagation** via manual calculus and chain rule derivatives[cite: 195].

**General Weight Update Rule:**
$$W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}$$

**Adam Optimization Formula (As Implemented):**
[cite_start]Adam combines the benefits of Momentum (moving average of gradients) and RMSProp (moving average of squared gradients)[cite: 301, 307]:
1.  $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (Momentum)
2.  $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (Velocity)
3.  $W = W - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

*This project was developed for the AI Optimization Techniques course at YTU CE (Computer Engineering).*
