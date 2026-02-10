import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# ==========================
# 1. VERİ YÜKLEME VE HAZIRLIK
# ==========================
print("Veri setleri yükleniyor...")
try:
    df_train = pd.read_csv('training.csv')
    df_test = pd.read_csv('test.csv')
except:
    print("HATA: 'training.csv' veya 'test.csv' bulunamadı.")
    exit()

# Etiketleri Hazırla (Zaten -1 ve 1 ise sadece boyutunu ayarla)
y_train = df_train['etiket'].values.reshape(-1, 1)
y_test = df_test['etiket'].values.reshape(-1, 1)

# ==========================
# 2. VEKTÖRLEŞTİRME (TF-IDF ve BERT)
# ==========================
def get_vectors(model_type):
    print(f"\n--> {model_type.upper()} ile vektörleştiriliyor...")
    
    questions_tr = ["query: " + str(q) for q in df_train['soru']]
    answers_tr = ["passage: " + str(a) for a in df_train['cevap']]
    questions_te = ["query: " + str(q) for q in df_test['soru']]
    answers_te = ["passage: " + str(a) for a in df_test['cevap']]

    if model_type == 'bert':
        # SentenceTransformer (Derin Öğrenme)
        model_st = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')
        q_tr = model_st.encode(questions_tr, show_progress_bar=True)
        a_tr = model_st.encode(answers_tr, show_progress_bar=True)
        q_te = model_st.encode(questions_te, show_progress_bar=True)
        a_te = model_st.encode(answers_te, show_progress_bar=True)
        
    elif model_type == 'tfidf':
        # TF-IDF (Klasik İstatistik)
        vectorizer = TfidfVectorizer(max_features=1024) 
        all_text = questions_tr + answers_tr + questions_te + answers_te
        vectorizer.fit(all_text)
        
        q_tr = vectorizer.transform(questions_tr).toarray()
        a_tr = vectorizer.transform(answers_tr).toarray()
        q_te = vectorizer.transform(questions_te).toarray()
        a_te = vectorizer.transform(answers_te).toarray()

    # Birleştirme ve Bias Ekleme
    X_train = np.hstack([q_tr, a_tr, np.ones((len(q_tr), 1))])
    X_test = np.hstack([q_te, a_te, np.ones((len(q_te), 1))])
    
    return X_train, X_test

# Verileri Hazırla
data_store = {}
data_store['bert'] = get_vectors('bert')
data_store['tfidf'] = get_vectors('tfidf')

# ==========================
# 3. MODEL VE OPTIMIZER SINIFLARI
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

class Optimizer:
    def __init__(self, method, lr=None):
        self.method = method
        self.t = 0 
        self.cache = {} 
        # Otomatik Hız Seçimi
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
# 4. DENEY FONKSİYONU (5 SEED ORTALAMALI)
# ==========================
def run_comparison():
    models_list = ['bert', 'tfidf'] # Listeyi burada tanımlıyoruz
    optimizers = ['gd', 'sgd', 'adam']
    seeds = [42, 10, 2024, 7, 99] # 5 Farklı Seed
    epochs = 100
    batch_size = 32
    
    # Sonuçları saklayacak yapı
    results = {m: {o: {'loss': [], 'acc': []} for o in optimizers} for m in models_list}
    
    print("\n" + "="*60)
    print("DENEY BAŞLIYOR: 2 MODEL x 3 OPTIMIZER (5 SEED ORTALAMASI)")
    print("="*60)

    for model_name in models_list:
        X_tr, X_te = data_store[model_name]
        
        for opt_name in optimizers:
            print(f"▶ Model: {model_name.upper()} | Optimizer: {opt_name.upper()}...")
            
            # Seedlerin sonuçlarını tutacak geçici listeler
            all_losses = []
            all_accs = []
            
            for seed in seeds:
                np.random.seed(seed)
                model = TwoLayerMLP(X_tr.shape[1], hidden_dim=64)
                optimizer = Optimizer(opt_name) # Otomatik LR
                
                seed_losses = []
                seed_accs = []
                
                for epoch in range(epochs):
                    # Eğitim
                    indices = np.arange(X_tr.shape[0])
                    np.random.shuffle(indices)
                    X_shuf = X_tr[indices]
                    y_shuf = y_train[indices]
                    
                    if opt_name == 'gd':
                        y_pred = model.forward(X_tr)
                        dW1, dW2 = model.backward(X_tr, y_train, y_pred)
                        optimizer.step(model, dW1, dW2)
                    else:
                        for i in range(0, X_tr.shape[0], batch_size):
                            X_batch = X_shuf[i:i+batch_size]
                            y_batch = y_shuf[i:i+batch_size]
                            y_pred_b = model.forward(X_batch)
                            dW1, dW2 = model.backward(X_batch, y_batch, y_pred_b)
                            optimizer.step(model, dW1, dW2)
                    
                    # Test Kayıt
                    y_pred_te = model.forward(X_te)
                    preds = np.where(y_pred_te > 0, 1, -1)
                    acc = accuracy_score(y_test, preds)
                    
                    y_pred_tr = model.forward(X_tr)
                    loss = np.mean((y_train - y_pred_tr)**2)
                    
                    seed_losses.append(loss)
                    seed_accs.append(acc)
                
                all_losses.append(seed_losses)
                all_accs.append(seed_accs)
            
            # 5 Seed'in Ortalamasını Alıp Kaydet
            results[model_name][opt_name]['loss'] = np.mean(all_losses, axis=0)
            results[model_name][opt_name]['acc'] = np.mean(all_accs, axis=0)
            
            print(f"  ✓ Ortalama Son Acc: %{results[model_name][opt_name]['acc'][-1]*100:.2f}")

    return results, epochs

# ==========================
# 5. ÇALIŞTIRMA VE GRAFİKLER
# ==========================
results, epochs_count = run_comparison()

# [DÜZELTME] Modeller listesini grafik çizimi için tekrar tanımlıyoruz
models_for_plot = ['bert', 'tfidf']

# Grafik Çizimi: 2 Satır (Modeller) x 2 Sütun (Loss/Acc)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
colors = {'gd': 'tab:red', 'sgd': 'tab:green', 'adam': 'tab:blue'}

for i, model_name in enumerate(models_for_plot):
    # Sol Sütun: LOSS
    ax_loss = axes[i, 0]
    for opt in ['gd', 'sgd', 'adam']:
        ax_loss.plot(range(epochs_count), results[model_name][opt]['loss'], 
                     label=opt.upper(), color=colors[opt], linewidth=2)
    ax_loss.set_title(f"{model_name.upper()} - Eğitim Hatası (Loss)", fontweight='bold')
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Sağ Sütun: ACCURACY
    ax_acc = axes[i, 1]
    for opt in ['gd', 'sgd', 'adam']:
        ax_acc.plot(range(epochs_count), results[model_name][opt]['acc'], 
                    label=opt.upper(), color=colors[opt], linewidth=2)
    ax_acc.set_title(f"{model_name.upper()} - Test Başarısı (Accuracy)", fontweight='bold')
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_5seed_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ 5 Seed ortalamalı dev grafik kaydedildi: final_5seed_comparison.png")