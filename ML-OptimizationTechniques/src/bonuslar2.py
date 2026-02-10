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
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')

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
# 3. OPTIMIZER SINIFI (RMSprop Eklendi)
# ==========================
class Optimizer:
    def __init__(self, method, lr=None):
        self.method = method
        self.t = 0 
        self.cache = {} 
        
        # Otomatik Hız Ayarları
        if lr is None:
            if method == 'adam': self.lr = 0.001   
            elif method == 'rmsprop': self.lr = 0.01  # [YENİ] RMSprop standardı
            elif method == 'adagrad': self.lr = 0.01   # Adagrad
            elif method == 'gd': self.lr = 0.01      
            elif method == 'sgd': self.lr = 0.1    
        else:
            self.lr = lr  
                    
    def step(self, model, dW1, dW2):
        params = [model.W1, model.W2]
        grads = [dW1, dW2]
        names = ['W1', 'W2']
        
        # --- GD ve SGD ---
        if self.method == 'gd' or self.method == 'sgd':
            for i in range(len(params)):
                params[i] -= self.lr * grads[i]

        # --- ADAM ---
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

        # --- ADAGRAD ---
        elif self.method == 'adagrad':
            eps = 1e-8
            for i, name in enumerate(names):
                g = grads[i]
                if name + '_cache' not in self.cache:
                    self.cache[name + '_cache'] = np.zeros_like(params[i])
                self.cache[name + '_cache'] += g ** 2
                params[i] -= self.lr * g / (np.sqrt(self.cache[name + '_cache']) + eps)

        # --- RMSPROP (YENİ EKLENEN KISIM) ---
        elif self.method == 'rmsprop':
            # RMSprop: Moving Average of Squared Gradients
            beta, eps = 0.9, 1e-8
            for i, name in enumerate(names):
                g = grads[i]
                
                # Cache başlatma (v: squared gradient average)
                if name + '_v' not in self.cache:
                    self.cache[name + '_v'] = np.zeros_like(params[i])
                
                v = self.cache[name + '_v']
                
                # Formül: v = beta * v + (1 - beta) * g^2
                v = beta * v + (1 - beta) * (g ** 2)
                self.cache[name + '_v'] = v
                
                # Güncelleme
                params[i] -= self.lr * g / (np.sqrt(v) + eps)
        
        model.W1, model.W2 = params[0], params[1] 

# ==========================
# 4. DENEY FONKSİYONU
# ==========================
def run_experiments():
    # 5 Algoritma: GD, SGD, Adam, Adagrad, RMSprop
    optimizers = ['gd', 'sgd', 'adam', 'adagrad', 'rmsprop']
    seeds = [42, 10, 2024, 7, 99]
    epochs = 100 
    batch_size = 32
    
    history = {opt: {"loss": [], "acc": [], "time": []} for opt in optimizers}
    
    for opt in optimizers:
        print(f"\n▶ Optimizer: {opt.upper()} başlıyor...")
        
        all_losses = []
        all_accs = []
        all_times = []
        
        for seed_idx, seed in enumerate(seeds):
            np.random.seed(seed)
            model = TwoLayerMLP(X_train.shape[1], hidden_dim=64)
            optimizer = Optimizer(opt)
            
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
                    dW1, dW2 = model.backward(X_train, y_train, y_pred)
                    optimizer.step(model, dW1, dW2)
                else: 
                    # Diğerleri mini-batch çalışır
                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW1, dW2)
                
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
print("MLP OPTIMIZER KARŞILAŞTIRMASI (RMSprop Dahil)")
print("=" * 50)
history, epochs_count = run_experiments()

# ==========================
# 5. GRAFİKLER (3x2 Grid - 5 Algoritma Rahat Sığar)
# ==========================
# 5 renk tanımlayalım
colors = {'adam': 'tab:green', 'adagrad': 'tab:red', 'rmsprop': 'tab:purple'}
labels = {opt: opt.upper() for opt in colors}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. EPOCH vs LOSS
ax = axes[0, 0]
for opt in colors:
    ax.plot(range(epochs_count), history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Epoch vs Training Loss (MSE)", fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 2. EPOCH vs ACCURACY
ax = axes[0, 1]
for opt in colors:
    ax.plot(range(epochs_count), history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Epoch vs Test Accuracy", fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. TIME vs LOSS
ax = axes[1, 0]
for opt in colors:
    ax.plot(history[opt]['time'], history[opt]['loss'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Time (s) vs Training Loss", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# 4. TIME vs ACCURACY
ax = axes[1, 1]
for opt in colors:
    ax.plot(history[opt]['time'], history[opt]['acc'], label=labels[opt], color=colors[opt], linewidth=2)
ax.set_title("Time (s) vs Test Accuracy", fontweight='bold')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_with_rmsprop.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Grafik kaydedildi: optimizer_with_rmsprop.png")
