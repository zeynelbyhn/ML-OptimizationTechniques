import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE  # t-SNE için gerekli

# ---------------------------------------------------------
# 1. VERİ YÜKLEME VE HAZIRLIK
# ---------------------------------------------------------
print("1. Veriler yükleniyor...")
# Dosya eşleşmeleri (Sizin yüklemenize göre)
df_train = pd.read_csv('test.csv')      # Gerçek Eğitim Seti
df_test = pd.read_csv('training.csv')   # Gerçek Test Seti

# Etiketleri (N, 1) boyutuna getir
y_train = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)

# Metinleri hazırla
train_qs = ["query: " + q for q in df_train['soru']]
train_as = ["passage: " + a for a in df_train['cevap']]
test_qs = ["query: " + q for q in df_test['soru']]
test_as = ["passage: " + a for a in df_test['cevap']]

# ---------------------------------------------------------
# 2. EMBEDDING MODELLERİ
# ---------------------------------------------------------
def get_embeddings(model_name, qs, ans):
    print(f"   Model yükleniyor: {model_name} ...")
    model = SentenceTransformer(model_name)
    
    print("   Metinler vektörleştiriliyor...")
    q_vec = model.encode(qs, show_progress_bar=False)
    a_vec = model.encode(ans, show_progress_bar=False)
    
    # Concat: [Soru, Cevap]
    concat = np.concatenate([q_vec, a_vec], axis=1)
    
    # Bias terimi ekle: [Soru, Cevap, 1]
    return np.hstack([concat, np.ones((concat.shape[0], 1))])

print("\n2. Temsil Modelleri Hazırlanıyor...")

# Model 1: Mevcut Model
X_train_m1 = get_embeddings('ytu-ce-cosmos/turkish-e5-large', train_qs, train_as)
X_test_m1 = get_embeddings('ytu-ce-cosmos/turkish-e5-large', test_qs, test_as)

# Model 2: Karşılaştırma Modeli (Multilingual MiniLM)
X_train_m2 = get_embeddings('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', train_qs, train_as)
X_test_m2 = get_embeddings('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', test_qs, test_as)

# ---------------------------------------------------------
# 3. GELİŞMİŞ OPTİMİZASYON SINIFI
# ---------------------------------------------------------
class NeuralOptimizer:
    def __init__(self, params, lr=0.01, method='adam'):
        self.params = params # Parametre referansları listesi
        self.lr = lr
        self.method = method
        self.t = 0
        self.state = {}
        
        # State başlatma
        for i, p in enumerate(params):
            self.state[i] = {'m': np.zeros_like(p), 'v': np.zeros_like(p)}
            
    def step(self, grads):
        self.t += 1
        eps = 1e-8
        
        for i, (p, g) in enumerate(zip(self.params, grads)):
            s = self.state[i]
            
            if self.method == 'gd' or self.method == 'sgd':
                p -= self.lr * g
                
            elif self.method == 'rmsprop':
                rho = 0.9
                s['v'] = rho * s['v'] + (1 - rho) * g**2
                p -= self.lr * g / (np.sqrt(s['v']) + eps)
                
            elif self.method == 'adagrad':
                s['v'] += g**2
                p -= self.lr * g / (np.sqrt(s['v']) + eps)
                
            elif self.method == 'adam':
                beta1, beta2 = 0.9, 0.999
                s['m'] = beta1 * s['m'] + (1 - beta1) * g
                s['v'] = beta2 * s['v'] + (1 - beta2) * g**2
                
                m_hat = s['m'] / (1 - beta1**self.t)
                v_hat = s['v'] / (1 - beta2**self.t)
                
                p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

# ---------------------------------------------------------
# 4. MİMARİLER: Single Layer & MLP
# ---------------------------------------------------------
class SingleLayer:
    def __init__(self, input_dim, seed=None): # Seed eklendi
        if seed is not None:
            np.random.seed(seed)
        self.w = np.random.randn(input_dim, 1) * 0.05 # Başlangıç ağırlığı biraz büyütüldü görselleştirme için
        self.params = [self.w]
        
    def forward(self, X):
        return np.tanh(X @ self.w)
    
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        grad = (2/N) * X.T @ ((y_pred - y_true) * (1 - y_pred**2))
        return [grad]

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.params = [self.W1, self.W2]
        
    def forward(self, X):
        self.z1 = X @ self.W1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2
        self.a2 = np.tanh(self.z2)
        return self.a2
        
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        dL_dy = 2 * (y_pred - y_true) / N
        delta2 = dL_dy * (1 - y_pred**2)
        dW2 = self.a1.T @ delta2
        delta1 = (delta2 @ self.W2.T) * (1 - self.a1**2)
        dW1 = X.T @ delta1
        return [dW1, dW2]

# Genel Eğitim Fonksiyonu (Bonuslar için)
def run_experiment(model_type, X_tr, y_tr, X_te, y_te, opt_method, epochs=50, lr=0.001):
    input_dim = X_tr.shape[1]
    if model_type == 'single':
        model = SingleLayer(input_dim)
    else:
        model = TwoLayerMLP(input_dim)
    optimizer = NeuralOptimizer(model.params, lr=lr, method=opt_method)
    acc_history = []
    for epoch in range(epochs):
        if opt_method == 'sgd':
            indices = np.arange(X_tr.shape[0])
            np.random.shuffle(indices)
            for i in indices:
                X_b, y_b = X_tr[i:i+1], y_tr[i:i+1]
                pred = model.forward(X_b)
                grads = model.backward(X_b, y_b, pred)
                optimizer.step(grads)
        else:
            pred = model.forward(X_tr)
            grads = model.backward(X_tr, y_tr, pred)
            optimizer.step(grads)
        test_pred = model.forward(X_te)
        acc = accuracy_score(np.sign(y_te), np.sign(test_pred))
        acc_history.append(acc)
    return acc_history

# ---------------------------------------------------------
# 5. BONUS DENEYLERİ ÇALIŞTIRMA (MEVCUT KISIM)
# ---------------------------------------------------------
print("\n--- BONUS 1: Eğitim Kümesi Büyüklüğü Etkisi ---")
train_sizes = [20, 40, 60, 80, 100]
size_accuracies = []
for size in train_sizes:
    X_sub = X_train_m1[:size]
    y_sub = y_train[:size]
    repeats = [run_experiment('single', X_sub, y_sub, X_test_m1, y_test, 'adam')[-1] for _ in range(3)]
    avg_acc = np.mean(repeats)
    size_accuracies.append(avg_acc)
    print(f"Size: {size}, Acc: {avg_acc:.2f}")

print("\n--- BONUS 2: Optimizasyon Algoritmaları ---")
opt_methods = ['gd', 'sgd', 'adam', 'rmsprop', 'adagrad']
opt_histories = {}
for opt in opt_methods:
    runs = [run_experiment('single', X_train_m1, y_train, X_test_m1, y_test, opt) for _ in range(3)]
    opt_histories[opt] = np.mean(runs, axis=0)
    print(f"{opt} tamamlandı.")

print("\n--- BONUS 3: Model Karşılaştırması ---")
res_m1 = run_experiment('single', X_train_m1, y_train, X_test_m1, y_test, 'adam')
res_m2 = run_experiment('single', X_train_m2, y_train, X_test_m2, y_test, 'adam')
model_accs = [res_m1[-1], res_m2[-1]]

print("\n--- BONUS 4: Mimari Karşılaştırması (MLP vs Single) ---")
res_mlp = run_experiment('mlp', X_train_m1, y_train, X_test_m1, y_test, 'adam')
arch_accs = [res_m1[-1], res_mlp[-1]]

# ---------------------------------------------------------
# 6. MEVCUT GÖRSELLEŞTİRME
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].plot(train_sizes, size_accuracies, marker='o', color='purple')
axs[0, 0].set_title('Eğitim Kümesi Boyutu vs Başarı')
axs[0, 0].set_xlabel('Örnek Sayısı')
axs[0, 0].set_ylabel('Test Accuracy')
axs[0, 0].grid(True)

for opt in opt_methods:
    axs[0, 1].plot(opt_histories[opt], label=opt)
axs[0, 1].set_title('Optimizasyon Yöntemleri')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].bar(['Turkish-E5', 'MiniLM (Multi)'], model_accs, color=['#1f77b4', '#ff7f0e'])
axs[1, 0].set_title('Dil Modeli Performansı')

axs[1, 1].bar(['Single Layer', '2-Layer MLP'], arch_accs, color=['#2ca02c', '#d62728'])
axs[1, 1].set_title('Mimari Karşılaştırması')
plt.tight_layout()
plt.show()

# =========================================================
# YENİ EKLENEN KISIM: PART B - YÖRÜNGE GÖRSELLEŞTİRME
# =========================================================
print("\n" + "="*60)
print("BÖLÜM B: OPTİMİZASYON YÖRÜNGELERİNİN t-SNE İLE GÖRSELLEŞTİRİLMESİ")
print("="*60)

seeds = [42, 10, 2023, 99, 7] # 5 farklı başlangıç değeri
traj_epochs = 40 # Yörünge için epoch sayısı
input_dim = X_train_m1.shape[1]

# Çizim için büyük bir figür açalım
plt.figure(figsize=(20, 12))

for plot_idx, opt_method in enumerate(opt_methods):
    print(f"Yörüngeler hesaplanıyor: {opt_method.upper()} ...")
    
    all_weights_for_tsne = [] # t-SNE için tüm ağırlıkları burada toplayacağız
    trajectories_indices = [] # Her seed'in başlangıç-bitiş indekslerini tutar
    
    current_idx = 0
    
    # 5 Farklı Seed İçin Döngü
    for seed in seeds:
        # Modeli belirli seed ile başlat
        model_traj = SingleLayer(input_dim, seed=seed)
        optimizer_traj = NeuralOptimizer(model_traj.params, lr=0.001, method=opt_method)
        
        trajectory = []
        # Başlangıç ağırlığını kaydet
        trajectory.append(model_traj.w.flatten().copy())
        
        # Eğitimi Başlat
        for ep in range(traj_epochs):
            if opt_method == 'sgd':
                indices = np.arange(X_train_m1.shape[0])
                np.random.shuffle(indices)
                for i in indices:
                    X_b, y_b = X_train_m1[i:i+1], y_train[i:i+1]
                    pred = model_traj.forward(X_b)
                    grads = model_traj.backward(X_b, y_b, pred)
                    optimizer_traj.step(grads)
            else:
                pred = model_traj.forward(X_train_m1)
                grads = model_traj.backward(X_train_m1, y_train, pred)
                optimizer_traj.step(grads)
            
            # Her epoch sonu ağırlığı kaydet
            trajectory.append(model_traj.w.flatten().copy())
        
        # Bu seed'in yörüngesini ana listeye ekle
        all_weights_for_tsne.extend(trajectory)
        
        # Bu seed'in kaç adım sürdüğünü kaydet (Görselleştirme için)
        traj_len = len(trajectory)
        trajectories_indices.append((current_idx, current_idx + traj_len, seed))
        current_idx += traj_len

    # --- t-SNE UYGULAMA ---
    # Tüm seedlerin tüm adımlarını tek bir matriste birleştirip 2 boyuta indiriyoruz
    combined_data = np.array(all_weights_for_tsne)
    
    # Perplexity veri sayısına göre ayarlanmalı (min 5, max 50)
    perp = min(30, len(combined_data) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    reduced_data = tsne.fit_transform(combined_data)
    
    # --- ÇİZİM ---
    ax = plt.subplot(2, 3, plot_idx+1)
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for i, (start, end, seed) in enumerate(trajectories_indices):
        # İlgili parçayı al
        traj_2d = reduced_data[start:end]
        
        # Çizgi çiz
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], marker='.', markersize=2, 
                color=colors[i], label=f'Seed {seed}', alpha=0.6, linewidth=1)
        
        # Başlangıç (Büyük Yuvarlak)
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker='o', s=80, color=colors[i], edgecolors='k', zorder=5)
        
        # Bitiş (Büyük Çarpı)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='X', s=100, color=colors[i], edgecolors='k', zorder=5)

    ax.set_title(f"Optimizer: {opt_method.upper()}")
    if plot_idx == 0: # Sadece ilkinde legend göster ki kalabalık olmasın
        ax.legend(fontsize='x-small')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- GRAFİK YORUMLAMA REHBERİ ---")
print("1. GD (Gradient Descent): Çizgiler genellikle pürüzsüz ve direkt hedefe yöneliktir. Zikzak yapmaz.")
print("2. SGD: Çizgiler çok 'gürültülü' ve zikzaklıdır. Rastgele örneklemden dolayı titreyerek ilerler.")
print("3. Adam: Çizgiler genellikle GD'den daha uzun adımlıdır (hızlı gider) ve momentumdan dolayı kavisli olabilir.")
print("4. Yuvarlak (O) noktalar başlangıç, Çarpı (X) noktalar bitişi gösterir.")
print("5. Eğer tüm renkler (farklı seed'ler) aynı bölgede toplanıyorsa, model kararlı bir çözüme ulaşıyor demektir.")