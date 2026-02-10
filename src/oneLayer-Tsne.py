import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

# ==========================
# 1. VERİ HAZIRLIĞI
# ==========================
print("Veri setleri yükleniyor...")
try:
    df_train = pd.read_csv('training.csv')
    df_test = pd.read_csv('test.csv')
except:
    print("HATA: Dosyalar bulunamadı.")
    exit()

print("Model yükleniyor...")
model_st = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

def get_vectorized_data(df):
    questions = ["query: " + str(q) for q in df['soru']]
    answers = ["passage: " + str(a) for a in df['cevap']]
    
    q_emb = model_st.encode(questions, show_progress_bar=True)
    a_emb = model_st.encode(answers, show_progress_bar=True)
    
    concat_vec = np.concatenate([q_emb, a_emb], axis=1)
    bias = np.ones((concat_vec.shape[0], 1))
    final_vec = np.hstack([concat_vec, bias])
    
    return final_vec

print("Veriler vektörleştiriliyor...")
X_train = get_vectorized_data(df_train)
X_test = get_vectorized_data(df_test)

# Etiketleri -1 ve 1 yapıyoruz (Tanh için)
y_train = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

input_dim = X_train.shape[1]
print(f"Veri Hazır! X_train: {X_train.shape}")

# ==========================
# 2. MODEL SINIFI
# ==========================
class OneLayerModel:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim, 1) * np.sqrt(1/input_dim)

    def forward(self, X):
        return np.tanh(X @ self.W)

    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        dW = X.T @ ((y_pred - y_true) * (1 - y_pred**2)) / N
        return dW

# ==========================
# 3. OPTIMIZER SINIFI
# ==========================
class Optimizer:
    def __init__(self, method, lr):
        self.method = method
        self.lr = lr
        self.cache = {}
        self.t = 0

    def step(self, model, dW):
        if self.method in ['gd', 'sgd']:
            model.W -= self.lr * dW
        
        elif self.method == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if 'm' not in self.cache:
                self.cache['m'] = np.zeros_like(model.W)
                self.cache['v'] = np.zeros_like(model.W)

            m = self.cache['m']
            v = self.cache['v']

            m = beta1 * m + (1 - beta1) * dW
            v = beta2 * v + (1 - beta2) * (dW**2)

            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)

            model.W -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            self.cache['m'] = m
            self.cache['v'] = v

# ==========================
# 4. DENEY FONKSİYONU (YÖRÜNGE KAYDI)
# ==========================
def run_trajectory_experiments():
    optimizers = {
        'gd':   0.01,    # Görsellik için yüksek hız
        'sgd':  0.1,   # Görsellik için orta hız
        'adam': 0.001   # Standart hız
    }
    
    seeds = [42, 10, 2024, 7, 99] # 5 Farklı Başlangıç Noktası
    epochs = 50 
    batch_size = 32
    
    # Ağırlık geçmişini saklamak için yapı:
    # {optimizer: {seed: [w_epoch_0, w_epoch_1, ...]}}
    weight_trajectories = {opt: {s: [] for s in seeds} for opt in optimizers}
    
    print("\n" + "="*50)
    print("YÖRÜNGE ANALİZİ BAŞLIYOR (Trajectory Tracking)")
    print("="*50)
    
    for opt, lr in optimizers.items():
        print(f"▶ Optimizer: {opt.upper()} hesaplanıyor...")
        
        for seed in seeds:
            np.random.seed(seed)
            model = OneLayerModel(input_dim)
            optimizer = Optimizer(opt, lr)
            
            # Başlangıç ağırlıklarını kaydet
            weight_trajectories[opt][seed].append(model.W.flatten().copy())
            
            for epoch in range(epochs):
                if opt == 'gd':
                    y_pred = model.forward(X_train)
                    dW = model.backward(X_train, y_train, y_pred)
                    optimizer.step(model, dW)
                else: 
                    # Shuffle
                    indices = np.arange(X_train.shape[0])
                    np.random.shuffle(indices)
                    X_shuf = X_train[indices]
                    y_shuf = y_train[indices]
                    
                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW)
                
                # Her epoch sonunda güncel ağırlıkları kaydet
                weight_trajectories[opt][seed].append(model.W.flatten().copy())
                
    return weight_trajectories, epochs

# ==========================
# 5. T-SNE GÖRSELLEŞTİRME
# ==========================
def plot_tsne_trajectories(weight_data, epochs):
    print("\n--> t-SNE ile Boyut İndirgeme Yapılıyor (Bu işlem biraz sürebilir)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    optimizers = ['gd', 'sgd', 'adam']
    
    # 5 Seed için 5 farklı renk
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] 
    
    for i, opt in enumerate(optimizers):
        ax = axes[i]
        
        # 1. Veriyi Hazırla: Bu optimizer'ın tüm seed ve epoch verilerini tek bir matrise yığ
        all_weights_for_opt = []
        seed_lengths = []
        
        for seed in weight_data[opt]:
            traj = np.array(weight_data[opt][seed]) 
            all_weights_for_opt.append(traj)
            seed_lengths.append(len(traj))
            
        # Matrisi birleştir (Stack)
        combined_matrix = np.vstack(all_weights_for_opt)
        
        # 2. t-SNE Uygula
        # Perplexity düşük tutuyoruz çünkü yörünge takibi yerel bir yapı
        # init='pca' daha kararlı sonuç verir
        tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, init='pca', learning_rate='auto', random_state=42)
        embedded_2d = tsne.fit_transform(combined_matrix)
        
        # 3. Çizim (Parçalayıp tekrar çiz)
        start_idx = 0
        for s_idx, seed in enumerate(weight_data[opt]):
            length = seed_lengths[s_idx]
            end_idx = start_idx + length
            
            # Bu seed'e ait 2D koordinatlar
            trajectory_2d = embedded_2d[start_idx:end_idx]
            
            # Yörüngeyi Çiz (Oklar veya Çizgi)
            ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                    marker='.', markersize=3, linestyle='-', linewidth=1.5, alpha=0.8, 
                    color=colors[s_idx], label=f'Seed {seed}')
            
            # Başlangıç Noktası (Yıldız)
            ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], marker='*', s=200, color=colors[s_idx], edgecolors='black', zorder=10)
            
            # Bitiş Noktası (Büyük Daire)
            ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], marker='o', s=120, color=colors[s_idx], edgecolors='black', zorder=10)
            
            start_idx = end_idx
            
        ax.set_title(f"{opt.upper()} Optimizasyon Yörüngeleri", fontweight='bold', fontsize=14)
        ax.set_xlabel("t-SNE Boyut 1")
        ax.set_ylabel("t-SNE Boyut 2")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(title="Başlangıçlar") 

    plt.tight_layout()
    plt.savefig('optimizer_trajectories_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Yörünge grafiği kaydedildi: optimizer_trajectories_tsne.png")

# Çalıştır
trajectories, total_epochs = run_trajectory_experiments()
plot_tsne_trajectories(trajectories, total_epochs)