# Neural Network Optimization from Scratch (NumPy Implementation) 🚀

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Core_Logic-013243?style=for-the-badge&logo=numpy)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

Bu proje, PyTorch veya TensorFlow gibi hazır derin öğrenme kütüphaneleri kullanılmadan, **tamamen NumPy kullanılarak sıfırdan** oluşturulmuş bir Yapay Sinir Ağı (MLP) ve çeşitli optimizasyon algoritmalarının kapsamlı bir analizidir.

Proje; veri üretiminden (LLM), vektörleştirmeye (Embeddings), geri yayılım (backpropagation) matematiğinden optimizasyon yörüngelerinin görselleştirilmesine (t-SNE) kadar uçtan uca bir yapay zeka mühendisliği çalışmasıdır.

## 🎯 Projenin Amacı

* **Matematiksel Derinlik:** Hazır fonksiyonlar (`model.fit()`) yerine, yapay zekanın temelindeki matematiksel işlemleri (türev, zincir kuralı, matris çarpımları) manuel olarak kodlayarak anlamak.
* **Optimizasyon Kıyaslaması:** Gradient Descent (GD), Stochastic Gradient Descent (SGD), Adam, AdaGrad ve RMSProp algoritmalarının performanslarını, hızlarını ve kararlılıklarını karşılaştırmak.
* **Görselleştirme:** Algoritmaların "Loss Landscape" (Hata Yüzeyi) üzerindeki hareketlerini t-SNE ile 2 boyuta indirgeyerek analiz etmek.

---

## 🛠️ Kullanılan Teknolojiler ve Yöntemler

Bu projede aşağıdaki adımlar "from-scratch" (sıfırdan) prensibiyle uygulanmıştır:

### 1. Veri Üretimi (Synthetic Data Generation)
* **Model:** Yerel olarak çalışan **Gemma-9B** modeli (Ollama üzerinden).
* **Yöntem:** Regresyon tabanlı bir sınıflandırma problemi için soru-cevap çiftleri üretildi.
* **Veri Seti:** Hatalı cevaplar (-1) ve doğru cevaplar (+1) olarak etiketlenmiş özgün Türkçe veri seti.

### 2. Veri Temsili (Semantic Embeddings)
* Kelime bazlı (TF-IDF) ve anlamsal bazlı (BERT/Transformer) yaklaşımlar kıyaslandı.
* **Model:** `ytu-ce-cosmos/turkish-e5-large` kullanılarak metinler 1024 boyutlu vektör uzayına taşındı.

### 3. Model Mimarisi (NumPy Only)
* **TwoLayerMLP:**
    * Input Layer: 2049 nöron (Soru + Cevap + Bias)
    * Hidden Layer: 64 nöron (Tanh aktivasyonu)
    * Output Layer: 1 nöron (Tanh aktivasyonu)
* **RecursiveMLP:** Dinamik katman sayısı için özyinelemeli (recursive) bir yapı kuruldu.

---

## 📊 Algoritma Karşılaştırmaları

Eğitim sürecinde (100 Epoch) elde edilen sonuçlara göre optimizasyon algoritmalarının karakteristikleri:

| Algoritma | Hız (Convergence) | Stabilite | Test Başarısı (Accuracy) | Karakteristik |
|-----------|-------------------|-----------|--------------------------|---------------|
| **GD** | 🔴 Yavaş          | 🟢 Çok Yüksek | 🟡 Düşük (~0.62) | Tüm veriyi tek seferde işler, zig-zag yapmaz ama çok yavaştır. |
| **SGD** | 🟡 Orta           | 🔴 Düşük | 🟢 İyi (~0.80) | Mini-batch (32) kullandığı için gürültülü (zig-zaglı) ilerler. |
| **Adam** | 🟢 **Çok Hızlı** | 🟢 **Yüksek** | 🌟 **Mükemmel (~0.98)** | Momentum ve Adaptive Learning Rate sayesinde en optimal çözümdür. |

---

## 📈 Sonuçlar ve Görselleştirmeler

### 1. Optimizasyon Yörüngeleri (t-SNE Analizi)
Aşağıdaki görselde, farklı algoritmaların global minimuma giden yolda nasıl hareket ettiği görülmektedir.
* **Adam:** Hedefe en kısa ve kararlı yoldan gider.
* **SGD:** Hedef etrafında salınım (oscillation) yapar.

*(Buraya projenizdeki t-SNE görselini -tsne_trajectory.png- ekleyin)*
`![t-SNE Trajectories](images/tsne_trajectory.png)`

### 2. Loss & Accuracy Grafikleri
Eğitim süresince hata (Loss) düşüş hızları ve doğruluk (Accuracy) artışları:

*(Buraya projenizdeki Loss/Accuracy grafiklerini ekleyin)*
`![Loss Graph](images/loss_graph.png)`

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/Neural-Network-Optimization-From-Scratch.git](https://github.com/KULLANICI_ADINIZ/Neural-Network-Optimization-From-Scratch.git)
    cd Neural-Network-Optimization-From-Scratch
    ```

2.  **Gereksinimleri Yükleyin:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn sentence-transformers
    ```

3.  **Modeli Eğitin:**
    ```bash
    python main.py
    ```

---

## 🧠 Teori: Nasıl Çalışıyor?

Model, **Geri Yayılım (Backpropagation)** algoritmasını manuel türev hesaplamalarıyla uygular.

**Ağırlık Güncelleme Kuralı (Genel):**
$$W_{yeni} = W_{eski} - \eta \cdot \frac{\partial L}{\partial W}$$

**Adam Optimizasyonu Formülü (Kod İçinde Uygulanan):**
Adam algoritması, gradyanların hareketli ortalamasını (Momentum) ve karelerinin hareketli ortalamasını (RMSProp) birleştirir:
1.  $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$ (Momentum)
2.  $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$ (Hız)
3.  $W = W - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

---

## 📜 Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakabilirsiniz.

---

*Bu proje, YTU CE (Computer Engineering) kapsamındaki Yapay Zeka Optimizasyon Teknikleri dersi için hazırlanmıştır.*
