import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

# ==========================
# 1. VERİ HAZIRLIĞI (OVERFITTING İÇİN AZALTILDI)
# ==========================
print("Veri setleri yükleniyor...")
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')

# [DEĞİŞİKLİK 1]: Veri setini küçültüyoruz (Sadece 50 örnek!)
# Az veri = Hızlı Ezberleme (Overfitting)
df_train = df_train.head(50) 

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

print(f"Veri Hazır! X_train: {X_train.shape} (Overfitting için bilerek azaltıldı)")

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
        else: self.lr = lr  
                    
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
# 4. DENEY FONKSİYONU (OVERFITTING AYARLARI)
# ==========================
def run_experiments():
    optimizers = ['gd', 'sgd', 'adam']
    # Tek seed yeterli, overfitting'i görmek istiyoruz
    seeds = [42] 
    
    # [DEĞİŞİKLİK 2]: Epoch sayısını çok artırıyoruz (Ezberlesin diye)
    epochs = 300 
    batch_size = 32
    
    history = {opt: {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": [], "time": []} for opt in optimizers}
    
    for opt in optimizers:
        print(f"\n▶ Optimizer: {opt.upper()} başlıyor...")
        
        # [DEĞİŞİKLİK 3]: Hızları biraz artırıyoruz ki hemen ezberlesin
        if opt == 'gd': lr = 0.5
        elif opt == 'sgd': lr = 0.1
        else: lr = 0.001
            
        print(f"  -> Learning Rate: {lr}")
        
        all_train_losses = []
        all_test_losses = [] # Test loss'u da takip edeceğiz (Overfitting kanıtı)
        all_train_accs = []
        all_test_accs = []
        all_times = []
        
        for seed in seeds:
            np.random.seed(seed)
            # [DEĞİŞİKLİK 4]: Hidden Dim'i devasa yapıyoruz (Kapasiteyi artır)
            # 64 yerine 1024 yapıyoruz.
            model = TwoLayerMLP(X_train.shape[1], hidden_dim=1024) 
            optimizer = Optimizer(opt, lr=lr)
            
            seed_train_losses = []
            seed_test_losses = []
            seed_train_accs = []
            seed_test_accs = []
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
                    for i in range(0, X_train.shape[0], batch_size):
                        X_batch = X_shuf[i:i+batch_size]
                        y_batch = y_shuf[i:i+batch_size]
                        y_pred_batch = model.forward(X_batch)
                        dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                        optimizer.step(model, dW1, dW2)
                
                elapsed = time.time() - start_time
                
                # --- TEST SONUÇLARI (Genelleme Başarısı) ---
                y_pred_test = model.forward(X_test)
                preds_test = np.where(y_pred_test > 0, 1, -1)
                acc_test = accuracy_score(y_test, preds_test)
                loss_test = np.mean((y_test - y_pred_test) ** 2)
                
                # --- TRAIN SONUÇLARI (Ezber Başarısı) ---
                y_pred_train = model.forward(X_train)
                preds_train = np.where(y_pred_train > 0, 1, -1)
                acc_train = accuracy_score(y_train, preds_train)
                loss_train = np.mean((y_train - y_pred_train) ** 2)
                
                seed_train_losses.append(loss_train)
                seed_test_losses.append(loss_test)
                seed_train_accs.append(acc_train)
                seed_test_accs.append(acc_test)
                seed_times.append(elapsed)
            
            all_train_losses.append(seed_train_losses)
            all_test_losses.append(seed_test_losses)
            all_train_accs.append(seed_train_accs)
            all_test_accs.append(seed_test_accs)
            all_times.append(seed_times)
            
        history[opt]["train_loss"] = np.mean(all_train_losses, axis=0)
        history[opt]["test_loss"] = np.mean(all_test_losses, axis=0)
        history[opt]["train_acc"] = np.mean(all_train_accs, axis=0)
        history[opt]["test_acc"] = np.mean(all_test_accs, axis=0)
        history[opt]["time"] = np.mean(all_times, axis=0)
        
        print(f"  ✓ Train Acc: %{history[opt]['train_acc'][-1]*100:.2f} | Test Acc: %{history[opt]['test_acc'][-1]*100:.2f}")

    return history, epochs

# Deneyi Çalıştır
print("=" * 50)
print("OVERFITTING DENEYİ BAŞLIYOR")
print("=" * 50)
history, epochs_count = run_experiments()

# ==========================
# 5. OVERFITTING GRAFİKLERİ (Train vs Test Farkı)
# ==========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
optimizers = ['gd', 'sgd', 'adam']

for i, opt in enumerate(optimizers):
    ax = axes[i]
    # Eğitim Başarısı (Ezber) -> Çizgi Düz
    ax.plot(range(epochs_count), history[opt]['train_acc'], label='Train Acc (Ezber)', color='green', linewidth=2)
    # Test Başarısı (Gerçek) -> Çizgi Kesikli
    ax.plot(range(epochs_count), history[opt]['test_acc'], label='Test Acc (Gerçek)', color='red', linestyle='--', linewidth=2)
    
    ax.set_title(f"{opt.upper()} - Overfitting Analizi", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_demonstration.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Overfitting grafiği kaydedildi: overfitting_demonstration.png")