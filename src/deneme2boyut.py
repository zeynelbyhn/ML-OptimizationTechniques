
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

# Etiketleri -1 ve 1 yapıyoruz
y_train = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)

print(f"Veri Hazır! X_train: {X_train.shape}")

# ==========================
# 2. MODEL SINIFI
# ==========================
class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=64):
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim    
        scale_w1 = np.sqrt(1 / input_dim)
        scale_w2 = np.sqrt(1 / hidden_dim)      
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale_w1
        self.W2 = np.random.randn(hidden_dim, 1) * scale_w2
        
    def forward(self, X):
        self.z1 = X @ self.W1            
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2    
        self.a2 = np.tanh(self.z2)      
        return self.a2
       
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        dy_pred = (y_pred - y_true) / N
        delta2 = dy_pred * (1 - self.a2**2)
        dW2 = self.a1.T @ delta2
        delta1 = (delta2 @ self.W2.T) * (1 - self.a1**2)
        dW1 = X.T @ delta1
        return dW1, dW2

# ==========================
# 3. OPTIMIZER SINIFI
# ==========================
class Optimizer:
    def __init__(self, method, lr=None):
        self.method = method
        self.t = 0 
        self.cache = {} 
        
        if lr is None:
            if method == 'adam': self.lr = 0.001   
            elif method == 'gd': self.lr = 0.01    # Görsellik için yüksek hız
            elif method == 'sgd': self.lr = 0.1   # Görsellik için orta hız
        else:
            self.lr = lr  
                    
    def step(self, model, dW1, dW2):
        params = [model.W1, model.W2]
        grads = [dW1, dW2]
        names = ['W1', 'W2']
        
        if self.method == 'gd' or self.method == 'sgd':
            for i in range(len(params)):
                params[i] -= self.lr * grads[i]

        elif self.method == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i, name in enumerate(names):
                g = grads[i] 
                if name + '_m' not in self.cache:
                    self.cache[name + '_m'] = np.zeros_like(params[i])
                    self.cache[name + '_v'] = np.zeros_like(params[i])
                m = self.cache[name + '_m']
                v = self.cache[name + '_v']
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                self.cache[name + '_m'] = m
                self.cache[name + '_v'] = v
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)
                params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        
        model.W1, model.W2 = params[0], params[1] 

# ==========================
# 4. DENEY FONKSİYONU (AĞIRLIKLARI KAYDEDEN)
# ==========================
def run_trajectory_experiments():
    optimizers = ['gd', 'sgd', 'adam']
    seeds = [42, 10, 2024, 7, 99] # 5 Farklı Başlangıç Noktası
    epochs = 50 # Yörüngeyi görmek için 50 adım yeterli
    batch_size = 32
    
    # Ağırlık geçmişini saklamak için yapı:
    # {optimizer: {seed: [w_epoch_0, w_epoch_1, ...]}}
    weight_trajectories = {opt: {s: [] for s in seeds} for opt in optimizers}
    
    print("\n" + "="*50)
    print("YÖRÜNGE ANALİZİ BAŞLIYOR (Trajectory Tracking)")
    print("="*50)
    
    for opt in optimizers:
        print(f"▶ Optimizer: {opt.upper()} hesaplanıyor...")
        
        for seed in seeds:
            np.random.seed(seed)
            model = TwoLayerMLP(X_train.shape[1], hidden_dim=64)
            optimizer = Optimizer(opt)
            
            # Başlangıç ağırlıklarını kaydet
            # Flatten yaparak tek bir vektör haline getiriyoruz (W1 + W2)
            initial_w = np.concatenate([model.W1.flatten(), model.W2.flatten()])
            weight_trajectories[opt][seed].append(initial_w)
            
            for epoch in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuf = X_train[indices]
                y_shuf = y_train[indices]
                
                if opt == 'gd':
                    y_pred = model.forward(X_train)
                    dW1, dW2 = model.backward(X_train, y_train, y_pred)
                    optimizer.step(model, dW1, dW2)
                else: 
                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW1, dW2)
                
                # Her epoch sonunda güncel ağırlıkları kaydet
                current_w = np.concatenate([model.W1.flatten(), model.W2.flatten()])
                weight_trajectories[opt][seed].append(current_w)
                
    return weight_trajectories, epochs
# ==========================
# 5. T-SNE GÖRSELLEŞTİRME (HEPSİ BİR ARADA)
# ==========================
def plot_tsne_trajectories(weight_data, epochs):
    print("\n--> t-SNE ile Boyut İndirgeme Yapılıyor (Bu işlem biraz sürebilir)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    optimizers = ['gd', 'sgd', 'adam']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'] # 5 Seed için 5 Renk
    
    for i, opt in enumerate(optimizers):
        ax = axes[i]
        
        # 1. Veriyi Hazırla: Bu optimizer'ın tüm seed ve epoch verilerini tek bir matrise yığ
        # Amaç: t-SNE hepsini aynı uzayda görsün.
        all_weights_for_opt = []
        seed_lengths = []
        
        for seed in weight_data[opt]:
            traj = np.array(weight_data[opt][seed]) # Shape: (Epochs+1, Total_Params)
            all_weights_for_opt.append(traj)
            seed_lengths.append(len(traj))
            
        # Matrisi birleştir (Stack)
        combined_matrix = np.vstack(all_weights_for_opt)
        
        # 2. t-SNE Uygula
        # Perplexity düşük tutuyoruz çünkü yörünge takibi yerel bir yapı
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
                    marker='.', markersize=2, linestyle='-', linewidth=1, alpha=0.7, 
                    color=colors[s_idx], label=f'Seed {seed}')
            
            # Başlangıç Noktası (Yıldız)
            ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], marker='*', s=150, color=colors[s_idx], edgecolors='black', zorder=10)
            
            # Bitiş Noktası (Büyük Daire)
            ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], marker='o', s=100, color=colors[s_idx], edgecolors='black', zorder=10)
            
            start_idx = end_idx
            
        ax.set_title(f"{opt.upper()} Optimizasyon Yörüngeleri", fontweight='bold')
        ax.set_xlabel("t-SNE Boyut 1")
        ax.set_ylabel("t-SNE Boyut 2")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(title="Başlangıçlar") # Sadece ilkinde legend olsun

    plt.tight_layout()
    plt.savefig('optimizer_trajectories_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ Yörünge grafiği kaydedildi: optimizer_trajectories_tsne.png")

# Çalıştır
trajectories, total_epochs = run_trajectory_experiments()
plot_tsne_trajectories(trajectories, total_epochs)