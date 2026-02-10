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

y_train = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)

print(f"Veri Hazır! X_train: {X_train.shape}")

# ==========================
# 2. RECURSIVE MLP MODELİ (DİNAMİK KATMANLI)
# ==========================
class RecursiveMLP:
    def __init__(self, layer_sizes):

        self.weights = []
        self.layer_sizes = layer_sizes
        self.activations = {} 
        
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            scale = np.sqrt(1 / n_in)
            self.weights.append(np.random.randn(n_in, n_out) * scale)
            
    def forward(self, X):
        return self._forward_recursive(X, 0)

    def _forward_recursive(self, current_input, layer_idx):
        if layer_idx == len(self.weights):
            return current_input
        
        self.activations[layer_idx] = current_input
        
        z = current_input @ self.weights[layer_idx]
        a = np.tanh(z)
        
        return self._forward_recursive(a, layer_idx + 1) 

    # --- RECURSIVE BACKWARD ---
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        dy_pred = (y_pred - y_true) / N
        
        # Son katman hatası
        last_delta = dy_pred * (1 - y_pred**2)
        
        grads = [None] * len(self.weights)
        
        # Sondan başa doğru özyineleme
        self._backward_recursive(last_delta, len(self.weights) - 1, grads)
        
        return grads

    def _backward_recursive(self, current_delta, layer_idx, grads_list):
        if layer_idx < 0:
            return

        prev_activation = self.activations[layer_idx]
        
        # dW hesapla
        dW = prev_activation.T @ current_delta
        grads_list[layer_idx] = dW 
        
        # Hatayı bir geriye taşı
        if layer_idx > 0:
            error_propagated = current_delta @ self.weights[layer_idx].T
            prev_delta = error_propagated * (1 - prev_activation**2)
            self._backward_recursive(prev_delta, layer_idx - 1, grads_list)

# ==========================
# 3. OPTIMIZER SINIFI (SADECE GD, SGD, ADAM)
# ==========================
class Optimizer:
    def __init__(self, method, lr=None):
        self.method = method
        self.t = 0 
        self.cache = {} 
        
        # Otomatik Hız Ayarları
        if lr is None:
            if method == 'adam': self.lr = 0.001   
            elif method == 'gd': self.lr = 0.01     # Hızlı GD
            elif method == 'sgd': self.lr = 0.1   # Hızlı SGD
        else:
            self.lr = lr  
    
    def step(self, model, grads_list):
        # GD ve SGD
        if self.method == 'gd' or self.method == 'sgd':
            for i in range(len(model.weights)):
                model.weights[i] -= self.lr * grads_list[i]

        # ADAM
        elif self.method == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            
            for i in range(len(model.weights)):
                g = grads_list[i]
                name = str(i)
                
                if name + '_m' not in self.cache:
                    self.cache[name + '_m'] = np.zeros_like(model.weights[i])
                    self.cache[name + '_v'] = np.zeros_like(model.weights[i])
                
                m = self.cache[name + '_m']
                v = self.cache[name + '_v']
                
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                
                self.cache[name + '_m'] = m
                self.cache[name + '_v'] = v
                
                m_hat = m / (1 - beta1 ** self.t)
                v_hat = v / (1 - beta2 ** self.t)
                
                model.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

# ==========================
# 4. DENEY FONKSİYONU
# ==========================
def run_experiments():
    optimizers = ['gd', 'sgd', 'adam']
    seeds = [42, 10, 2024, 7, 99]
    epochs = 100 
    batch_size = 32
    
    input_dim = X_train.shape[1]
    layer_structure = [input_dim, 128, 64, 32, 1] 
    
    history = {opt: {"loss": [], "acc": [], "time": []} for opt in optimizers}
    
    for opt in optimizers:
        print(f"\n▶ Optimizer: {opt.upper()} başlıyor...")
        
        all_losses = []
        all_accs = []
        all_times = []
        
        for seed_idx, seed in enumerate(seeds):
            np.random.seed(seed)
            model = RecursiveMLP(layer_structure)
            optimizer = Optimizer(opt)
            
            if seed_idx == 0: print(f"  -> LR: {optimizer.lr} | Yapı: {layer_structure}")
            
            seed_losses = []
            seed_accs = []
            seed_times = []
            
            start_time = time.time()
            
            for epoch in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuf = X_train[indices]
                y_shuf = y_train[indices]
                
                if opt == 'gd':
                    y_pred = model.forward(X_train)
                    grads = model.backward(X_train, y_train, y_pred)
                    optimizer.step(model, grads)
                else: 
                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        grads = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, grads)
                
                elapsed = time.time() - start_time
                y_pred_test = model.forward(X_test)
                preds_binary = np.where(y_pred_test > 0, 1, -1)
                acc = accuracy_score(y_test, preds_binary)
                y_pred_train = model.forward(X_train)
                loss = np.mean((y_train - y_pred_train) ** 2)
                
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

# Deneyi Çalıştır
print("=" * 50)
print("RECURSIVE MLP: GD - SGD - ADAM")
print("=" * 50)
history, epochs_count = run_experiments()

# ==========================
# 5. GRAFİKLER (2x2 Grid)
# ==========================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
colors = {'gd': 'tab:blue', 'sgd': 'tab:orange', 'adam': 'tab:green'}
labels = {'gd': 'GD', 'sgd': 'SGD', 'adam': 'Adam'}

# 1. EPOCH vs LOSS
ax = axes[0, 0]
for opt in history:
    ax.plot(range(epochs_count), history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Epoch vs Training Loss (MSE)", fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. EPOCH vs ACCURACY
ax = axes[0, 1]
for opt in history:
    ax.plot(range(epochs_count), history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Epoch vs Test Accuracy", fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. TIME vs LOSS
ax = axes[1, 0]
for opt in history:
    ax.plot(history[opt]['time'], history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Time (s) vs Training Loss", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. TIME vs ACCURACY
ax = axes[1, 1]
for opt in history:
    ax.plot(history[opt]['time'], history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Time (s) vs Test Accuracy", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('recursive_gd_sgd_adam.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Grafik kaydedildi: recursive_gd_sgd_adam.png")