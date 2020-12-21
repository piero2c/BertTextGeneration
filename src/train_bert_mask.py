import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from tqdm import tqdm

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

def criterion(y_true, y_pred):
    return tf.math.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred[batch_input['input_ids'] == _MASK])
    )

def load_first(fpath, n):
    with open(fpath, encoding='utf-8') as f_iter:
        if n == -1:
            sample = [line.rstrip() for line in tqdm(f_iter.readlines())]
        else:
            sample = [next(f_iter).rstrip() for _ in tqdm(range(n))]
    return sample

# Interface CLI
parser = argparse.ArgumentParser('Treina modelo BERT-MASK.')
parser.add_argument('--nb_instances', type=int,
                    help='Limita o número de instâncias de treinamento (prepared_data/bert/tr_*). Se --nb_instance=-1, utiliza todos os dados. [-1]',
                    default=-1)
parser.add_argument('--max_title_len', type=int, default=12, help='Número de tokens máximos para um título. '\
                    'Por padrão, utiliza 12 tokens.')
parser.add_argument('--max_review_len', type=int, default=58, help='Número de tokens máximos para um review. '\
                    'Por padrão, utiliza 58 tokens.')
parser.add_argument('--nb_epochs', type=int, help='Número de épocas de treinamento. [1]', default=1)
parser.add_argument('--lr', type=float, help='Learning rate do otimizador ADAM. [5e-6]', default=5e-6)
parser.add_argument('--nb_grad_acc', type=int, help='Número de etapas de `gradient_accumulation` por batch [4]', default=4)
parser.add_argument('--batch_size', type=int, help='Tamanho do batch (que será multiplicado por `nb_grad_acc`) [12]', default=12)
parser.add_argument('--log_each', type=int, help='Número de iterações para que uma mensagem de evolução do treinamento seja exibida [1]', default=1)
parser.add_argument('--save_each', type=int, help='Salva modelo a cada `--save-each` iterações de treinamento [100]', default=10_000)


if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = Path(__file__).absolute().parent.parent

    print(f'Carregando Tokenizador BERTimbau...')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True)

    # Adiciona token especial que usaremos para delimitar o final de cada sentença
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused1]']
    })

    # Obtém os ids dos tokens especiais
    _CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')

    print('Carregando dados...')

    reviews = load_first(root_dir / 'prepared_data/bert/tr_reviews.txt', n=args.nb_instances)
    titles = load_first(root_dir / 'prepared_data/bert/tr_titles.txt', n=args.nb_instances)
    labels = [int(label) for label in load_first(root_dir / 'prepared_data/bert/tr_labels.txt', n=args.nb_instances)]

    print(f'Iniciando tokenização. Isso pode levar alguns minutos e consumir bastante memória (fique de olho!)')

    # Aplica tokenizador (isso pode levar alguns minutos e irá consumir uma quantidade significativa de memória)
    # Precisamos passar os títulos no campo `text_pair` para garantir que os embeddings do tipo de sentença  (sentence_type)
    # do bert sejam passados corretamente para o modelo
    dataset = tokenizer(reviews, text_pair=titles, padding=True,
                        max_length=args.max_review_len + args.max_title_len + 2,
                        truncation=True)

    # Cria tf.dataset dos dados de treino
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(dataset),
        np.array(labels, dtype='int32')
    ))

    # Carrega modelo pré-treinado
    print(f'Carregando modelo BERTimbau...')
    bert = TFAutoModelForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                  from_pt=True)

    # Instância otimizador e variáveis auxiliares
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    nb_grad_acc, batch_size = args.nb_grad_acc, args.batch_size
    nb_opt_steps = int(len(train_dataset)/(nb_grad_acc*batch_size))
    loss_hist = []

    for _ in range(args.nb_epochs):
        # Instância iterador da época
        batch_iterator = train_dataset.shuffle(1000).batch(batch_size).__iter__()

        # Itera nos batches
        for i in tqdm(range(nb_opt_steps)):
            with tf.GradientTape() as tape:
                loss_value = 0
                
                # Acumula gradientes
                for _ in range(nb_grad_acc):
                    batch_input, batch_labels = next(batch_iterator)
                    output = bert(batch_input)[0]
                    loss_value += criterion(batch_labels, output)
            
                # Guarda valores da loss no histórico
                loss_value = loss_value/nb_grad_acc
                loss_hist.append(loss_value)

                # Calcula gradientes e avança o otimizador
                grads = tape.gradient(loss_value, bert.trainable_weights)
                optimizer.apply_gradients(zip(grads, bert.trainable_weights))

            # Exibe mensagem de progresso
            if (i+1) % args.log_each == 0:
                print(f'Iter {i}/{nb_opt_steps}: {loss_value}')
            
            if (i+1) % args.save_each == 0:
                bert.save_pretrained(root_dir / f'models/bert_mask_ckpt-{i}')
                print(f'Saved model in ./bert_mask_ckpt-{i}')

    print(f'Treinamento finalizado. Salvando resultado models/ft_bert_mask')
    bert.save_pretrained(str(root_dir / 'models/ft_bert_mask'))