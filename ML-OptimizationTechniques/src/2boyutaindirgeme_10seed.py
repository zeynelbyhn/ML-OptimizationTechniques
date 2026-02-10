import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
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


print(f"Veri Hazır! X_train: {X_train.shape}s")

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
            elif method == 'gd': self.lr = 0.01      
            elif method == 'sgd': self.lr = 0.1    
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
# 4. DENEY FONKSİYONU (10 SEED İLE)
# ==========================
def run_trajectory_experiments():
    optimizers = ['gd', 'sgd', 'adam']
    
    # [GÜNCELLEME]: 10 Farklı Seed
    seeds = [42, 10, 2024, 70, 99, 1, 2, 3, 4, 5] 
    
    epochs = 50 
    batch_size = 32
    
    weight_trajectories = {opt: {s: [] for s in seeds} for opt in optimizers}
    
    print("\n" + "="*50)
    print("YÖRÜNGE ANALİZİ BAŞLIYOR (10 SEED)")
    print("="*50)
    
    for opt in optimizers:
        print(f"▶ Optimizer: {opt.upper()} hesaplanıyor...")
        
        for seed in seeds:
            np.random.seed(seed)
            model = TwoLayerMLP(X_train.shape[1], hidden_dim=64)
            optimizer = Optimizer(opt)
            
            # Başlangıç ağırlıkları
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
                
                # Güncel ağırlıkları kaydet
                current_w = np.concatenate([model.W1.flatten(), model.W2.flatten()])
                weight_trajectories[opt][seed].append(current_w)
                
    return weight_trajectories, epochs, seeds

# ==========================
# 5. T-SNE GÖRSELLEŞTİRME (10 RENK İLE)
# ==========================
def plot_tsne_trajectories(weight_data, epochs, seeds):
    print("\n--> t-SNE ile Boyut İndirgeme Yapılıyor (Bu işlem biraz sürebilir)...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    optimizers = ['gd', 'sgd', 'adam']
    
    # [GÜNCELLEME]: 10 farklı renk paleti ("tab10" idealdir)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, opt in enumerate(optimizers):
        ax = axes[i]
        
        all_weights_for_opt = []
        seed_lengths = []
        
        for seed in weight_data[opt]:
            traj = np.array(weight_data[opt][seed]) 
            all_weights_for_opt.append(traj)
            seed_lengths.append(len(traj))
            
        combined_matrix = np.vstack(all_weights_for_opt)
        
        tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, init='pca', learning_rate='auto', random_state=42)
        embedded_2d = tsne.fit_transform(combined_matrix)
        
        start_idx = 0
        for s_idx, seed in enumerate(weight_data[opt]):
            length = seed_lengths[s_idx]
            end_idx = start_idx + length
            
            trajectory_2d = embedded_2d[start_idx:end_idx]
            
            # Yörünge Çizimi
            ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                    marker='.', markersize=2, linestyle='-', linewidth=1, alpha=0.7, 
                    color=colors[s_idx], label=f'Seed {seed}')
            
            # Başlangıç (Yıldız)
            ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], marker='*', s=120, color=colors[s_idx], edgecolors='black', zorder=10)
            
            # Bitiş (Daire)
            ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], marker='o', s=80, color=colors[s_idx], edgecolors='black', zorder=10)
            
            start_idx = end_idx
            
        ax.set_title(f"{opt.upper()} Optimizasyon Yörüngeleri", fontweight='bold')
        ax.set_xlabel("t-SNE Boyut 1")
        ax.set_ylabel("t-SNE Boyut 2")
        ax.grid(True, alpha=0.3)
        
        # Legend sadece ilk grafikte olsun (yer kaplamasın)
        if i == 0: 
            ax.legend(title="Başlangıçlar", fontsize='small', loc='upper left', bbox_to_anchor=(-0.1, 1))

    plt.tight_layout()
    plt.savefig('optimizer_trajectories_tsne_10seeds.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\n✓ 10 Seed'li Yörünge grafiği kaydedildi: optimizer_trajectories_tsne_10seeds.png")

# Çalıştır
trajectories, total_epochs, seed_list = run_trajectory_experiments()
plot_tsne_trajectories(trajectories, total_epochs, seed_list)