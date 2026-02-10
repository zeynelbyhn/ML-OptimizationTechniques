import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

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
# 2. EMBEDDING MODELLERİ (Bonus 3: Farklı Modeller)
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
# 3. GELİŞMİŞ OPTİMİZASYON SINIFI (Bonus 2)
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
# 4. MİMARİLER: Single Layer & MLP (Bonus 4)
# ---------------------------------------------------------
class SingleLayer:
    def __init__(self, input_dim):
        self.w = np.random.randn(input_dim, 1) * 0.01
        self.params = [self.w]
        
    def forward(self, X):
        return np.tanh(X @ self.w)
    
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        # Loss türevi (MSE): 2*(y_pred - y_true)
        # Tanh türevi: (1 - y_pred^2)
        grad = (2/N) * X.T @ ((y_pred - y_true) * (1 - y_pred**2))
        return [grad]

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim=64):
        # W1: Girdi -> Gizli, W2: Gizli -> Çıktı
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.params = [self.W1, self.W2]
        
    def forward(self, X):
        self.z1 = X @ self.W1
        self.a1 = np.tanh(self.z1) # Gizli katman aktivasyonu
        self.z2 = self.a1 @ self.W2
        self.a2 = np.tanh(self.z2) # Çıktı katmanı
        return self.a2
        
    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        # dL/dy_pred
        dL_dy = 2 * (y_pred - y_true) / N
        
        # Çıktı katmanı gradyanı
        delta2 = dL_dy * (1 - y_pred**2)
        dW2 = self.a1.T @ delta2
        
        # Gizli katman gradyanı (Backprop)
        delta1 = (delta2 @ self.W2.T) * (1 - self.a1**2)
        dW1 = X.T @ delta1
        
        return [dW1, dW2]

# Genel Eğitim Fonksiyonu
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
            
        # Test Başarısını Ölç
        test_pred = model.forward(X_te)
        acc = accuracy_score(np.sign(y_te), np.sign(test_pred))
        acc_history.append(acc)
        
    return acc_history

# ---------------------------------------------------------
# 5. DENEYLERİ ÇALIŞTIRMA
# ---------------------------------------------------------
print("\n--- BONUS 1: Eğitim Kümesi Büyüklüğü Etkisi ---")
train_sizes = [20, 40, 60, 80, 100]
size_accuracies = []

for size in train_sizes:
    # Veri setini kırp
    X_sub = X_train_m1[:size]
    y_sub = y_train[:size]
    
    # Her deneyi 3 kez tekrarla, ortalamasını al
    repeats = [run_experiment('single', X_sub, y_sub, X_test_m1, y_test, 'adam')[-1] for _ in range(3)]
    avg_acc = np.mean(repeats)
    size_accuracies.append(avg_acc)
    print(f"Size: {size}, Acc: {avg_acc:.2f}")

print("\n--- BONUS 2: Optimizasyon Algoritmaları ---")
opt_methods = ['gd', 'sgd', 'adam', 'rmsprop', 'adagrad']
opt_histories = {}

for opt in opt_methods:
    # 3 tekrar ortalaması
    runs = [run_experiment('single', X_train_m1, y_train, X_test_m1, y_test, opt) for _ in range(3)]
    opt_histories[opt] = np.mean(runs, axis=0)
    print(f"{opt} tamamlandı.")

print("\n--- BONUS 3: Model Karşılaştırması ---")
# İki modeli de Adam ile eğitip karşılaştıralım
res_m1 = run_experiment('single', X_train_m1, y_train, X_test_m1, y_test, 'adam')
res_m2 = run_experiment('single', X_train_m2, y_train, X_test_m2, y_test, 'adam')
model_accs = [res_m1[-1], res_m2[-1]]

print("\n--- BONUS 4: Mimari Karşılaştırması (MLP vs Single) ---")
res_mlp = run_experiment('mlp', X_train_m1, y_train, X_test_m1, y_test, 'adam')
arch_accs = [res_m1[-1], res_mlp[-1]]

# ---------------------------------------------------------
# 6. GÖRSELLEŞTİRME
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Grafik 1: Eğitim Büyüklüğü
axs[0, 0].plot(train_sizes, size_accuracies, marker='o', color='purple')
axs[0, 0].set_title('Eğitim Kümesi Boyutu vs Başarı')
axs[0, 0].set_xlabel('Örnek Sayısı')
axs[0, 0].set_ylabel('Test Accuracy')
axs[0, 0].grid(True)

# Grafik 2: Optimizasyonlar
for opt in opt_methods:
    axs[0, 1].plot(opt_histories[opt], label=opt)
axs[0, 1].set_title('Optimizasyon Yöntemleri')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Test Accuracy')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Grafik 3: Modeller
axs[1, 0].bar(['Turkish-E5', 'MiniLM (Multi)'], model_accs, color=['#1f77b4', '#ff7f0e'])
axs[1, 0].set_title('Dil Modeli Performansı')
axs[1, 0].set_ylim(0, 1.1)
for i, v in enumerate(model_accs):
    axs[1, 0].text(i, v + 0.02, f"{v:.2f}", ha='center')

# Grafik 4: Mimari
axs[1, 1].bar(['Single Layer', '2-Layer MLP'], arch_accs, color=['#2ca02c', '#d62728'])
axs[1, 1].set_title('Mimari Karşılaştırması')
axs[1, 1].set_ylim(0, 1.1)
for i, v in enumerate(arch_accs):
    axs[1, 1].text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.show()

print("Tüm bonus analizleri tamamlandı!")