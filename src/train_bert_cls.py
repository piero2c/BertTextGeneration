import os
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

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    pass

if __name__ == "__main__":
    root_dir = Path(__file__).absolute().parent.parent

    # Carrega tokenizador do bert treinado pela neuralmind
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', use_fast=True)

    # Adiciona um token especial que usaremos para delimitar o final da sentença
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused1]']
    })

    _CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')

    # Carrega datasets

    def load_first(fpath, n=1_000):
        with open(fpath, encoding='utf-8') as f_iter:
            sample = [next(f_iter).rstrip() for _ in range(n)]
        return sample

    tr_size = 400_000

    reviews = load_first(root_dir / 'prepared_data/bert/tr_reviews.txt', n=tr_size)
    titles = load_first(root_dir / 'prepared_data/bert/tr_titles.txt', n=tr_size)
    labels = [int(label) for label in load_first(root_dir / 'prepared_data/bert/tr_labels.txt', n=tr_size)]

    # Aplica tokenizador (isso pode levar alguns minutos e irá consumir uma quantidade significativa de memória)
    dataset = tokenizer(reviews, text_pair=titles, padding=True,
                        max_length=58 + 11 + 2,
                        truncation=True)

    # Testes
    # tokenizer.decode(dataset['input_ids'][35])
    # tokenizer.decode(dataset['input_ids'][36])

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
        load = True
        if load:
            ckpt = tf.train.Checkpoint(model=bert)
            bert.ckpt_manager = tf.train.CheckpointManager(ckpt, str(root_dir / 'bkp/checkpoint/'), max_to_keep=10)
            
            if bert.ckpt_manager.latest_checkpoint:
                print(f'Restoring: {bert.ckpt_manager.latest_checkpoint}')
                ckpt.restore(bert.ckpt_manager.latest_checkpoint).expect_partial()

    bert.save_pretrained(str(root_dir / 'models/ft_bert_cls'))

    trainer = TFTrainer(
        model=bert,
        args=training_args,
        train_dataset=train_dataset.shuffle(10_000),
    )

    # Realiza treinamento
    trainer.train()

    # Salva modelo
    trainer.save_model(str(root_dir / 'models/ft_bert_cls'))