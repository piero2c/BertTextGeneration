import argparse
import os
from tqdm import tqdm
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import transformers
transformers.logging.set_verbosity_info()

from transformers import (AutoTokenizer, TFAutoModelForMaskedLM,
                          TFAutoModelForSequenceClassification, AutoConfig,
                          TFTrainer, TFTrainingArguments)

from transformers.trainer_utils import EvaluationStrategy
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

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

def load_first(fpath, n):
        with open(fpath, encoding='utf-8') as f_iter:
            if n == -1:
                sample = [line.rstrip() for line in tqdm(f_iter.readlines())]
            else:
                sample = [next(f_iter).rstrip() for _ in tqdm(range(n))]
        return sample

# Interface CLI
parser = argparse.ArgumentParser('Treina modelo BERT-CLS.')
parser.add_argument('--nb_instances', type=int,
                    help='Limita o número de instâncias de treinamento (prepared_data/bert/tr_*). Se --nb_instance=-1, utiliza todos os dados. [-1]',
                    default=-1)
parser.add_argument('--max_title_len', type=int, default=12, help='Número de tokens máximos para um título. '\
                    'Por padrão, utiliza 12 tokens.')
parser.add_argument('--max_review_len', type=int, default=58, help='Número de tokens máximos para um review. '\
                    'Por padrão, utiliza 58 tokens.')
parser.add_argument('--nb_epochs', type=int, help='Número de épocas de treinamento. [1]', default=1)
parser.add_argument('--lr', type=float, help='Learning rate do otimizador ADAM. [5e-5]', default=5e-5)
parser.add_argument('--nb_grad_acc', type=int, help='Número de etapas de `gradient_accumulation` por batch [2]', default=2)
parser.add_argument('--batch_size', type=int, help='Tamanho do batch (que será multiplicado por `nb_grad_acc`) [10]', default=10)
parser.add_argument('--log_each', type=int, help='Número de iterações para que uma mensagem de evolução do treinamento seja exibida [10]', default=10)
parser.add_argument('--save_each', type=int, help='Salva modelo a cada `--save-each` iterações de treinamento [2000]', default=2_000)

if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = Path(__file__).absolute().parent.parent

    # Carrega tokenizador do bert treinado pela neuralmind
    print(f'Carregando Tokenizador BERTimbau...')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True)

    # Adiciona um token especial que usaremos para delimitar o final da sentença
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused1]']
    })

    # Obtém os ids dos tokens especiais
    _CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')

    # Carrega datasets
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

    # Define parâmetros de treinamento
    training_args = TFTrainingArguments(
        output_dir='./results2', 
        logging_dir='./logs2',       
        learning_rate=5e-5,  
        num_train_epochs=1,
        save_steps=2_000,
        logging_steps=10,
        per_device_train_batch_size=10,
        gradient_accumulation_steps=2,
        save_total_limit=10,
    )
    
    # Carrega BERT pré-treinado para o português
    with training_args.strategy.scope():
        bert = TFAutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                                    from_pt=True,
                                                                    num_labels=tokenizer.vocab_size)

    trainer = TFTrainer(
        model=bert,
        args=training_args,
        train_dataset=train_dataset.shuffle(10_000),
    )

    # Realiza treinamento
    trainer.train()

    # Salva modelo
    save_path = str(root_dir / 'models/ft_bert_cls')
    print(f'Treinamento finalizado. Salvando resultado em {save_path}')
    trainer.save_model(save_path)