import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


df_train = pd.read_csv('training.csv')     
df_test = pd.read_csv('test.csv')  

model = SentenceTransformer('ytu-ce-cosmos/turkish-e5-large')

def get_vectorized_data(df):
    questions = ["query: " + q for q in df['soru']]
    answers = ["passage: " + a for a in df['cevap']]
    
    q_emb = model.encode(questions)
    a_emb = model.encode(answers)
    
    concat_vec = np.concatenate([q_emb, a_emb], axis=1)

    bias = np.ones((concat_vec.shape[0], 1))
    final_vec = np.hstack([concat_vec, bias])
    
    return final_vec


X_train = get_vectorized_data(df_train)
X_test = get_vectorized_data(df_test)

