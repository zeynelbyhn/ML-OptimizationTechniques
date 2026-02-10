import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
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
# 2. MODEL SINIFI (Dinamik Hidden Dim)
# ==========================
class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim):
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
# 3. OPTIMIZER SINIFI (Adam Sabitlendi)
# ==========================
class Optimizer:
    def __init__(self, method='adam', lr=0.001):
        self.method = method
        self.lr = lr # Artık dışarıdan gelen LR'yi kabul ediyor
        self.t = 0 
        self.cache = {} 

    def step(self, model, dW1, dW2):
        params = [model.W1, model.W2]
        grads = [dW1, dW2]
        names = ['W1', 'W2']
        
        # Adam Optimizer
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
# 4. HİPERPARAMETRE TARAMASI (GRID SEARCH)
# ==========================
def run_grid_search():
    print("\n" + "="*50)
    print("HİPERPARAMETRE TARAMASI BAŞLIYOR (GRID SEARCH)")
    print("="*50)
    
    # Denenecek Parametreler
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_dims = [32, 64, 128]
    
    # Sonuçları saklayacak matris (Heatmap için)
    # Satırlar: Hidden Dims, Sütunlar: Learning Rates
    results_matrix = np.zeros((len(hidden_dims), len(learning_rates)))
    
    best_acc = 0
    best_params = {}
    
    epochs = 30 
    batch_size = 32
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    hidden_dims = [32, 64, 128]
    
    for i, h_dim in enumerate(hidden_dims):
        for j, lr in enumerate(learning_rates):
            
           
            
            # Modeli Başlat
            np.random.seed(42) # Adil karşılaştırma için sabit seed
            model = TwoLayerMLP(X_train.shape[1], hidden_dim=h_dim)
            optimizer = Optimizer(method='adam', lr=lr)
            
            # Hızlı Eğitim Döngüsü
            for epoch in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuf = X_train[indices]
                y_shuf = y_train[indices]
                
                for k in range(0, X_train.shape[0], batch_size):
                    X_batch = X_shuf[k:k+batch_size]
                    y_batch = y_shuf[k:k+batch_size]
                    y_pred_batch = model.forward(X_batch)
                    dW1, dW2 = model.backward(X_batch, y_batch, y_pred_batch)
                    optimizer.step(model, dW1, dW2)
            
            # Test Başarısını Ölç
            y_pred_test = model.forward(X_test)
            preds_binary = np.where(y_pred_test > 0, 1, -1)
            acc = accuracy_score(y_test, preds_binary)
            
            # Sonucu Kaydet
            results_matrix[i, j] = acc
            print(f"Acc: %{acc*100:.2f}")
            
            # En iyiyi takip et
            if acc > best_acc:
                best_acc = acc
                best_params = {'hidden_dim': h_dim, 'lr': lr}

    print("="*50)
    print(f"🏆 EN İYİ SONUÇ: Acc = %{best_acc*100:.2f}")
    print(f"   Ayarlar: Hidden Dim = {best_params['hidden_dim']}, LR = {best_params['lr']}")
    print("="*50)
    
    return results_matrix, learning_rates, hidden_dims

# ==========================
# 5. ÇALIŞTIRMA VE HEATMAP ÇİZİMİ
# ==========================
results_matrix, lrs, h_dims = run_grid_search()

plt.figure(figsize=(10, 8))
sns.heatmap(results_matrix, annot=True, fmt=".3f", cmap="viridis",
            xticklabels=lrs, yticklabels=h_dims)

plt.title("Hiperparametre Taraması: Hidden Size vs Learning Rate (Accuracy)", fontweight='bold')
plt.xlabel("Learning Rate")
plt.ylabel("Hidden Layer Size")
plt.savefig('hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Isı haritası kaydedildi: hyperparameter_heatmap.png")
