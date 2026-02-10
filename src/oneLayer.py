import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score

# ==========================
# 1) DATA VE EMBEDDING
# ==========================
df_train = pd.read_csv('training.csv')
df_test = pd.read_csv('test.csv')

print("Embedding modeli yükleniyor...")
model_st = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

def get_vectorized_data(df):
    q = ["query: " + str(x) for x in df['soru']]
    a = ["passage: " + str(x) for x in df['cevap']]
    q_emb = model_st.encode(q)
    a_emb = model_st.encode(a)

    x = np.concatenate([q_emb, a_emb], axis=1)
    bias = np.ones((x.shape[0],1))
    x = np.hstack([x,bias])
    return x

X_train = get_vectorized_data(df_train)
X_test  = get_vectorized_data(df_test)

y_train = df_train['etiket'].values.reshape(-1,1)
y_test  = df_test['etiket'].values.reshape(-1,1)

input_dim = X_train.shape[1]

# ==========================
# 2) ONE LAYER MODEL
# ==========================
class OneLayerModel:
    def __init__(self, input_dim):
        self.W = np.random.randn(input_dim,1) * np.sqrt(1/input_dim)

    def forward(self, X):
        return np.tanh(X @ self.W)

    def backward(self, X, y_true, y_pred):
        N = X.shape[0]
        dW = X.T @ ((y_pred - y_true)*(1 - y_pred**2)) / N
        return dW

# ==========================
# 3) OPTIMIZER
# ==========================
class Optimizer:
    def __init__(self, method, lr):
        self.method = method
        self.lr = lr
        self.cache = {}
        self.t = 0

    def step(self, model, dW):
        if self.method in ['gd','sgd']:
            model.W -= self.lr * dW
        
        elif self.method == 'adam':
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if 'm' not in self.cache:
                self.cache['m'] = np.zeros_like(model.W)
                self.cache['v'] = np.zeros_like(model.W)

            m = self.cache['m']
            v = self.cache['v']

            m = beta1*m + (1-beta1)*dW
            v = beta2*v + (1-beta2)*(dW**2)

            m_hat = m / (1-beta1**self.t)
            v_hat = v / (1-beta2**self.t)

            model.W -= self.lr * m_hat / (np.sqrt(v_hat)+eps)
            self.cache['m'] = m
            self.cache['v'] = v

# ==========================
# 4) TRAIN
# ==========================
def run():
    optimizers = {
        'gd':   0.01,
        'sgd':  0.1,
        'adam': 0.001
    }

    epochs = 100
    batch_size = 32

    hist = {opt:{'loss':[],'acc':[],'time':[]} for opt in optimizers}

    for opt,lr in optimizers.items():

        model = OneLayerModel(input_dim)
        optimizer = Optimizer(opt,lr)

        start_time = time.time()

        for ep in range(epochs):

            if opt=='gd':
                y_pred = model.forward(X_train)
                dW = model.backward(X_train,y_train,y_pred)
                optimizer.step(model,dW)

            else:
                idx = np.random.permutation(X_train.shape[0])
                Xs = X_train[idx]
                ys = y_train[idx]

                for i in range(0,X_train.shape[0],batch_size):
                    Xb = Xs[i:i+batch_size]
                    yb = ys[i:i+batch_size]
                    y_pred_batch = model.forward(Xb)
                    dW = model.backward(Xb,yb,y_pred_batch)
                    optimizer.step(model,dW)

            # ölç
            y_pred_train = model.forward(X_train)
            loss = np.mean((y_train-y_pred_train)**2)

            y_pred_test = model.forward(X_test)
            acc = accuracy_score(y_test,np.sign(y_pred_test))

            elapsed = time.time() - start_time

            hist[opt]['loss'].append(loss)
            hist[opt]['acc'].append(acc)
            hist[opt]['time'].append(elapsed)

    return hist,epochs


history,epochs = run()

# ==========================
# 5) GRAFİKLER
# ==========================

plt.figure(figsize=(13,9))

# 1) Epoch vs Loss
plt.subplot(2,2,1)
for opt in history:
    plt.plot(range(epochs),history[opt]['loss'],label=opt.upper())
plt.title("Epoch vs Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 2) Epoch vs Accuracy
plt.subplot(2,2,2)
for opt in history:
    plt.plot(range(epochs),history[opt]['acc'],label=opt.upper())
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# 3) Time vs Loss
plt.subplot(2,2,3)
for opt in history:
    plt.plot(history[opt]['time'],history[opt]['loss'],label=opt.upper())
plt.title("Time vs Loss")
plt.xlabel("Time (s)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 4) Time vs Accuracy
plt.subplot(2,2,4)
for opt in history:
    plt.plot(history[opt]['time'],history[opt]['acc'],label=opt.upper())
plt.title("Time vs Accuracy")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
