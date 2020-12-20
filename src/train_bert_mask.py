import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow import keras
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, TFAutoModelForMaskedLM, AutoConfig

# Ativa mixed-precision para economizar VRAM
try:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
except Exception as e:
    print(f'Precisão mista não habilitada (erro: {str(e)})')

# Ativa VRAM expansiva
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass

print(f'Carregando BERTimbau...')

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True)

# Adiciona token especial que usaremos para delimitar o final de cada sentença
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[unused1]']
})

# Obtém os ids dos tokens especiais
_CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')

reviews = [line.rstrip() for line in open('./reviews.txt', encoding='utf-8').readlines()][0:500_000]
titles = [line.rstrip() for line in open('./titles.txt', encoding='utf-8').readlines()][0:500_000]
labels = [int(label.rstrip()) for label in open('./labels.txt', encoding='utf-8').readlines()][0:500_000]

dataset = tokenizer(reviews, text_pair=titles, padding=True,
                    max_length=58 + 11 + 2,
                    truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dataset),
    np.array(labels, dtype='int32')
))

bert = TFAutoModelForMaskedLM.from_pretrained('./bert_finetuned')