import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

# ==========================
# 1. VERİ HAZIRLIĞI
# ==========================
print("Veri setleri yükleniyor...")
# Dosya yollarının doğru olduğundan emin olun
try:
    df_train = pd.read_csv('training.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("HATA: CSV dosyaları bulunamadı. Lütfen dosya yollarını kontrol edin.")
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
X_train_full = get_vectorized_data(df_train)
X_test = get_vectorized_data(df_test)

# Etiketleri sadece reshape yapıyoruz (Zaten -1 ve 1 oldukları için)
y_train_full = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)

print(f"Veri Hazır! X_train: {X_train_full.shape}")

# ==========================
# 2. MODEL SINIFI (Xavier Init)
# ==========================
class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=64):
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        
        # Xavier Initialization
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
# 3. OPTIMIZER SINIFI (Otomatik LR)
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
# 4. DENEY 1: OPTIMIZER KARŞILAŞTIRMASI
# ==========================
def run_optimizer_comparison():
    print("\n" + "="*40)
    print("DENEY 1: OPTIMIZER KARŞILAŞTIRMASI")
    print("="*40)
    
    optimizers = ['gd', 'sgd', 'adam']
    seeds = [42, 10, 2024, 7, 99]
    epochs = 100 
    batch_size = 32
    
    history = {opt: {"loss": [], "acc": [], "time": []} for opt in optimizers}
    
    for opt in optimizers:
        print(f"\n▶ Optimizer: {opt.upper()} başlıyor...")
        
        all_losses = []
        all_accs = []
        all_times = []
        
        for seed in seeds:
            np.random.seed(seed)
            model = TwoLayerMLP(X_train_full.shape[1], hidden_dim=64)
            optimizer = Optimizer(opt)
            
            seed_losses = []
            seed_accs = []
            seed_times = []
            
            start_time = time.time()
            
            for epoch in range(epochs):
                indices = np.arange(X_train_full.shape[0])
                np.random.shuffle(indices)
                X_shuf = X_train_full[indices]
                y_shuf = y_train_full[indices]
                
                if opt == 'gd':
                    y_pred = model.forward(X_train_full)
                    dW1, dW2 = model.backward(X_train_full, y_train_full, y_pred)
                    optimizer.step(model, dW1, dW2)
                else: 
                    for i in range(0, X_train_full.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW1, dW2)
                
                elapsed = time.time() - start_time
                y_pred_test = model.forward(X_test)
                preds_binary = np.where(y_pred_test > 0, 1, -1)
                acc = accuracy_score(y_test, preds_binary)
                y_pred_train = model.forward(X_train_full)
                loss = np.mean((y_train_full - y_pred_train) ** 2)
                
                seed_losses.append(loss)
                seed_accs.append(acc)
                seed_times.append(elapsed)
            
            all_losses.append(seed_losses)
            all_accs.append(seed_accs)
            all_times.append(seed_times)
            
        history[opt]["loss"] = np.mean(all_losses, axis=0)
        history[opt]["acc"] = np.mean(all_accs, axis=0)
        history[opt]["time"] = np.mean(all_times, axis=0)
        
        print(f"  ✓ Sonuç: Acc=%{history[opt]['acc'][-1]*100:.2f}")

    return history, epochs

# ==========================
# 5. DENEY 2: VERİ BOYUTU ETKİSİ (YENİ EKLENEN KISIM)
# ==========================
def run_size_experiments():
    print("\n" + "="*40)
    print("DENEY 2: EĞİTİM KÜMESİ BÜYÜKLÜĞÜ ANALİZİ")
    print("="*40)
    
    # Veri setinin %10, %25, %50, %75 ve %100'ünü deneyeceğiz
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    optimizers = ['gd', 'sgd', 'adam']
    
    # Sonuçları saklayacak sözlük
    size_results = {opt: [] for opt in optimizers}
    
    # Karıştırma (Shuffling) önemli, yoksa hep aynı sınıf gelebilir
    indices = np.arange(X_train_full.shape[0])
    np.random.seed(42) # Tekrarlanabilirlik için sabit seed
    np.random.shuffle(indices)
    
    X_shuffled = X_train_full[indices]
    y_shuffled = y_train_full[indices]
    
    epochs = 50 # Bu deney için 50 epoch yeterli olabilir
    batch_size = 32
    
    for opt in optimizers:
        print(f"\n▶ Optimizer: {opt.upper()} (Veri Boyutu Analizi)...")
        
        for frac in fractions:
            # Veri setini kesiyoruz (Slice)
            limit = int(X_shuffled.shape[0] * frac)
            X_sub = X_shuffled[:limit]
            y_sub = y_shuffled[:limit]
            
            # Modeli sıfırlıyoruz
            model = TwoLayerMLP(X_sub.shape[1], hidden_dim=64)
            optimizer = Optimizer(opt) # Otomatik LR kullanır
            
            # Hızlı eğitim döngüsü (Tek seed yeterli trendi görmek için)
            for epoch in range(epochs):
                if opt == 'gd':
                    y_pred = model.forward(X_sub)
                    dW1, dW2 = model.backward(X_sub, y_sub, y_pred)
                    optimizer.step(model, dW1, dW2)
                else:
                    # Mini-batch için karıştırma
                    sub_indices = np.arange(X_sub.shape[0])
                    np.random.shuffle(sub_indices)
                    X_sub_shuf = X_sub[sub_indices]
                    y_sub_shuf = y_sub[sub_indices]
                    
                    for i in range(0, X_sub.shape[0], batch_size):
                        X_batch = X_sub_shuf[i:i+batch_size]
                        y_batch = y_sub_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW1, dW2)
            
            # Eğitim bitince Test başarısını ölç
            y_pred_test = model.forward(X_test)
            preds_binary = np.where(y_pred_test > 0, 1, -1)
            final_acc = accuracy_score(y_test, preds_binary)
            
            size_results[opt].append(final_acc)
            print(f"  -> Veri: %{frac*100:3.0f} ({limit} örnek) | Acc: %{final_acc*100:.2f}")
            
    return fractions, size_results

# ==========================
# ÇALIŞTIRMA VE GRAFİKLER
# ==========================

# 1. Deneyi Çalıştır (Optimizer Karşılaştırması)
history, epochs_count = run_optimizer_comparison()

# 2. Deneyi Çalıştır (Veri Boyutu Analizi)
fractions, size_results = run_size_experiments()

# -----------------
# GRAFİK ÇİZİMİ
# -----------------
# Toplam 3 Satır olacak şekilde ayarlıyoruz
fig = plt.figure(figsize=(15, 18))

colors = {'gd': 'tab:blue', 'sgd': 'tab:orange', 'adam': 'tab:green'}
labels = {'gd': 'GD', 'sgd': 'SGD', 'adam': 'Adam'}

# --- GRAFIK 1 & 2: EPOCH ANALİZİ (Yan Yana) ---
ax1 = plt.subplot(3, 2, 1)
for opt in history:
    ax1.plot(range(epochs_count), history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax1.set_title("Epoch vs Eğitim Hatası (Loss)", fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 2, 2)
for opt in history:
    ax2.plot(range(epochs_count), history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax2.set_title("Epoch vs Test Başarısı", fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

# --- GRAFIK 3 & 4: ZAMAN ANALİZİ (Yan Yana) ---
ax3 = plt.subplot(3, 2, 3)
for opt in history:
    ax3.plot(history[opt]['time'], history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax3.set_title("Süre (sn) vs Eğitim Hatası", fontweight='bold')
ax3.set_xlabel("Süre (sn)")
ax3.set_ylabel("Loss")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 2, 4)
for opt in history:
    ax4.plot(history[opt]['time'], history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax4.set_title("Süre (sn) vs Test Başarısı", fontweight='bold')
ax4.set_xlabel("Süre (sn)")
ax4.set_ylabel("Accuracy")
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- GRAFIK 5: EĞİTİM KÜMESİ BÜYÜKLÜĞÜ ANALİZİ (Altta, Geniş) ---
ax5 = plt.subplot(3, 1, 3) # Alt satır komple kaplasın
x_labels = [f"%{int(f*100)}" for f in fractions]
for opt in size_results:
    ax5.plot(x_labels, size_results[opt], marker='o', linestyle='-', linewidth=2, markersize=8, label=labels[opt], color=colors[opt])

ax5.set_title("Eğitim Kümesi Büyüklüğünün Test Başarısına Etkisi", fontweight='bold', fontsize=14)
ax5.set_xlabel("Eğitim Kümesi Oranı (%)", fontsize=12)
ax5.set_ylabel("Test Başarısı (Accuracy)", fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig('full_analysis_report.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Tüm analizler tamamlandı ve 'full_analysis_report.png' olarak kaydedildi.")