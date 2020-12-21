import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

import lstm_model

from nltk.tokenize import word_tokenize

# Ativa memória expansiva para a primeira GPU
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass

# Verifica se o punkt está instalado na máquina
try:
    word_tokenize('Apenas um texto', language='portuguese')
except Exception as e:
    print(e)
    raise ValueError('Você precisa instalar o tokenizador punkt do NLTK. Veja como fazer isso em: https://www.nltk.org/data.html')

# Estruturas do modelo

# Carrega tokenizer do NLTK para pt
tokenizer = lambda x: word_tokenize(x, language='portuguese')

# Define parser
parser = argparse.ArgumentParser('Treina LSTM bidirecional para geração de títulos.')
parser.add_argument('--batch_size', type=int,
                    help='Tamanho do batch',
                    default=512)
parser.add_argument('--max_epochs', type=int,
                    help='Número máximo de épocas. O número efetivo de épocas depende dos parâmetros de early stopping',
                    default=30)
parser.add_argument('--patience', type=int,
                    help='Paciência (em número de épocas) do early stopping ',
                    default=5)
parser.add_argument('--emb_dim', type=int,
                    help='Dimensão da camada de embeddings [128]',
                    default=128)
parser.add_argument('--max_review_len', type=int,
                    help='Número de tokens máximos para uma review. '\
                         'Por padrão, utiliza 58 tokens (que corresponde ao quantil 0.9 da distribuição dos tamanhos)',
                    default=58)
parser.add_argument('--max_title_len', type=int,
                    help='Número de tokens máximos para uma review. '\
                         'Por padrão, utiliza 12 tokens (que corresponde ao quantil 0.9 da distribuição dos tamanhos)',
                    default=12)
parser.add_argument('--vocab_size', type=int,
                    help='Tamanho do vocabulário do modelo. [30 mil]',
                    default=30_000)
parser.add_argument('--uncased', action='store_true',
                    help='Aplica lower-case antes nos dados')
parser.add_argument('--prediction_nb_batches', type=int, default=20,
                    help='Número de batches usados na etapa de predição. Em caso de erros de memória nesta etapa, aumentar o valor desse parâmetro (20, por padrão). [20]')
args = parser.parse_args()


if __name__ == '__main__':
    root_dir = Path(__file__).absolute().parent.parent
    data_path = root_dir / 'prepared_data/lstm/'
    assert data_path.exists(), f'Caminho {data_path} não existe. Execute o script de preprocessamento.'
    
    # Carrega dados de treinamento
    tr_reviews = open(data_path / 'tr_reviews.txt', encoding='utf-8').readlines()
    tr_titles = open(data_path / 'tr_titles.txt', encoding='utf-8').readlines()
    te_reviews = open(data_path / 'te_reviews.txt', encoding='utf-8').readlines()
    te_titles = open(data_path / 'te_titles.txt', encoding='utf-8').readlines()

    # Inclui tags <s> e </s> 
    tr_reviews, tr_titles, te_reviews, te_titles = [[ ['<s>'] + tokenizer(text) + ['</s>'] for text in dataset]
                                                    for dataset in (tr_reviews, tr_titles, te_reviews, te_titles)]

    # Cria vocabulário a partir dos textos tokenizados
    text2idx = keras.preprocessing.text.Tokenizer(num_words=args.vocab_size, oov_token='<unk>', 
                                                  lower=args.uncased)
    text2idx.fit_on_texts(tr_reviews)

    # Converte tokens para ids
    tr_reviews_idx, tr_titles_idx, te_reviews_idx, te_titles_idx = [
        text2idx.texts_to_sequences(s) for s in [tr_reviews, tr_titles, te_reviews, te_titles]
    ]

    # Faz padding
    pad_x = lambda x: pad_sequences(x, maxlen=args.max_review_len, padding='pre')
    pad_y = lambda y: pad_sequences(y, maxlen=args.max_title_len, padding='post')

    tr_reviews_idx, te_reviews_idx = pad_x(tr_reviews_idx), pad_x(te_reviews_idx)
    tr_titles_idx, te_titles_idx = pad_y(tr_titles_idx), pad_y(te_titles_idx)

    # Cria modelo
    model = lstm_model.build_model(args.max_review_len, args.max_title_len, args.vocab_size, args.emb_dim)

    # Define otimizador, função de custo e callbacks
    optim = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optim, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    ]

    # Cria variável target (próxima palavra da sequência)
    tr_titles_idx_next = np.array([
        array[1:].tolist() + [0] for array in tr_titles_idx
    ])

    # Usa os primeiros 10% dos dados para validação
    val_break = int(len(tr_reviews_idx)*0.1)
    val_data = ([tr_reviews_idx[:val_break], tr_titles_idx[:val_break]], tr_titles_idx_next[:val_break])

    # Faz ajuste do modelo
    loss_hist = model.fit(x=[tr_reviews_idx[val_break:], tr_titles_idx[val_break:]], y=tr_titles_idx_next[val_break:],
                          batch_size=args.batch_size, epochs=args.max_epochs,
                          callbacks=callbacks, validation_data=val_data)
    
    weights_path = Path(__file__).absolute().parent.parent / 'models/lstm_weights/'
    if not weights_path.exists():
        weights_path.mkdir()

    model.save_weights(str(weights_path / 'lstm'))

    np.savetxt(str(weights_path / 'tr_loss.txt'), np.array(loss_hist.history['loss']), delimiter=",")
    np.savetxt(str(weights_path / 'val_loss.txt'), np.array(loss_hist.history['val_loss']), delimiter=",")

    print(f'Os pesos do modelo foram salvos em {weights_path}')

    print(f'Iniciando predição do modelo no conjunto de validação')

    # Tive que dividir a chamada abaixo em batches por conta de algum bug de memory leakage
    # no model.predict do keras. Diminuir o parâmetro `batch_size` do `model.predict` não funcionou e 
    # a solução abaixo foi o único jeito que achei de evitar OOM :(
    # Caso esteja ainda com esse problema, aumente o valor do argumento `--prediction_nb_batches`.
    # Leia sobre outras pessoas com o mesmo problema nas threads abaixo.
    # https://github.com/keras-team/keras/issues/13118
    # https://github.com/tensorflow/tensorflow/issues/33030
    try:
        nb_batches = args.prediction_nb_batches
        te_titles = []
        step = int(len(te_reviews_idx)/nb_batches) + 1
        for i in tqdm(range(0, nb_batches)):
            te_titles += lstm_model.write_titles(model, text2idx, te_reviews_idx[(i*step):((i+1)*step)],
                                                 max_title_len=args.max_title_len)
    except Exception as e:
        print(e)
        print('Erro na etapa de previsão. Em caso de falta de memória, aumente o valor de `--prediction_nb_batches`.')

    # Junta os tokens em um texto e remove tags
    te_titles = [
        ' '.join(titles).replace('<s>', '').replace('</s>', '') + '\n' for titles in te_titles
    ]

    # Salva os resultados
    open(root_dir / 'predictions/lstm.txt', 'w', encoding='utf-8').writelines(te_titles)
    print(f'Previsões foram salvas em {root_dir / "predictions/lstm.txt"}')
