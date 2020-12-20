from tqdm import tqdm
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_data(fpath, comma_separator=False):
    # Carregamento do arquivo
    fpath = Path(fpath)
    assert fpath.exists(), f'Arquivo {fpath} não encontrado.'
    df = pd.read_csv(fpath, sep=',' if comma_separator else ';')

    # Colunas relevantes para o modelo
    relevant_cols = ['review_text', 'review_title']
    assert all(col in df.columns for col in relevant_cols)
    
    # Filtra colunas relevantes
    df = df[relevant_cols]

    # Faz casting de tipos 
    df = df.astype('str')

    return df

def split_data(df, train_size, seed):
    # Separa os textos
    X, y = df.review_text.values.tolist(), df.review_title.values.tolist()

    # Permuta aleatoriamente linhas do DataFrame
    shuffled_idx = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(shuffled_idx)

    # Faz divisão em treino e teste
    X = np.array(X)[shuffled_idx]
    y = np.array(y)[shuffled_idx]
    p = int(len(X)*train_size)

    tr_X, tr_y = X[:p], y[:p]
    te_X, te_y = X[p:], y[p:]
    
    return tr_X, tr_y, te_X, te_y

def save_lstm_data(output_dir, tr_X, tr_y, te_X, te_y):
    output_dir = Path(output_dir)
    assert output_dir.is_dir(), f'Pasta {output_dir} não existe.'

    if not (output_dir / 'lstm').exists():
        (output_dir / 'lstm').mkdir()
    output_dir = output_dir / 'lstm'

    (tr_X, tr_y, te_X, te_y) = [[item + '\n' for item in d.tolist()] for d in [tr_X, tr_y, te_X, te_y]]

    open(output_dir / 'tr_reviews.txt', 'w', encoding='utf-8').writelines(tr_X)
    open(output_dir / 'tr_titles.txt', 'w', encoding='utf-8').writelines(tr_y)
    
    open(output_dir / 'te_reviews.txt', 'w', encoding='utf-8').writelines(te_X)
    open(output_dir / 'te_titles.txt', 'w', encoding='utf-8').writelines(te_y)

def prepare_titles_for_bert(tokenizer, y, max_title_len):
    # Obtém o índice dos tokens especiais
    _CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')

    # Adiciona o token final de final de sentença
    y = [yi + ' [unused1]' for yi in y]

    # Tokeniza os títulos
    tokenized_y = tokenizer(y, padding=True, max_length=max_title_len, truncation=True)

    # Obtém o tamanho de cada título
    lens = np.where(np.array(tokenized_y['input_ids']) == _SEP)[-1]

    # Gera um novo dataset
    new_y, new_labels = [], []

    for i, seq in enumerate(tqdm(tokenized_y['input_ids'])):
        # Remove o primeiro e o último token (`[CLS]` e `[SEP]`)
        non_special_tokens = seq[1:lens[i]]

        # Cria sequência para geração de texto autoregressiva
        new_y.append(tokenizer.batch_decode([
            non_special_tokens[:i] + [_MASK]
            for i in range(lens[i]-1)
        ]))

        # Guarda o índice do token
        new_labels.append([non_special_tokens[i] for i in range(lens[i]-1)])
    
    return new_y, new_labels

def prepare_and_save_bert_seqs(output_dir, tr_X, tr_y, te_X, te_y, bert_max_review_len, bert_max_title_len):
    output_dir = Path(output_dir)
    assert output_dir.is_dir(), f'Pasta {output_dir} não existe.'

    if not (output_dir / 'bert').exists():
        (output_dir / 'bert').mkdir()
    output_dir = output_dir / 'bert'

    # Carrega o tokenizer para o bertimbau
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    # Adiciona à lista de tokens especiais o token `[unused1]` que usaremos para delimitar o fim de um título
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused1]']
    })

    # Trunca reviews
    tr_X, te_X = [
        tokenizer.batch_decode(tokenizer(X.tolist(), add_special_tokens=False, padding=False, 
                                         max_length=bert_max_review_len, truncation=True)['input_ids']
        ) for X in (tr_X, te_X)]
    
    # Prepara títulos para a task de geração autoregressiva
    (tr_y, tr_labels), (te_y, te_labels) = [prepare_titles_for_bert(tokenizer, y, bert_max_title_len) for y in [tr_y, te_y]]

    # Salva resultado
    for X, y, labels in [(tr_X, tr_y, tr_labels), (te_X, te_y, te_labels)]:
        split = 'tr' if len(X) == len(tr_X) else 'te'

        with open(output_dir / f'{split}_reviews.txt', 'w', encoding='utf-8') as fp:
            for review_nb, review in enumerate(tqdm(X)):
                # Repete o mesmo review várias vezes para garantir que reviews.txt tenha o mesmo número
                # de linhas de titles.txt
                for title_variation in y[review_nb]:
                    fp.write(review + '\n')
                
        with open(output_dir / f'{split}_titles.txt', 'w', encoding='utf-8') as fp:
            for review_nb, review in enumerate(tqdm(X)):
                for title_variation in y[review_nb]:
                    fp.write(title_variation + '\n')

        with open(output_dir / f'{split}_labels.txt', 'w', encoding='utf-8') as fp:
            for review_nb, _ in enumerate(tqdm(X)):
                for label in labels[review_nb]:
                    fp.write(str(label) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Realiza preprocessamento dos dados.')
    parser.add_argument('-i', type=str, help='Caminho ao arquivo de entrada `B2W-Reviews01.csv` (grande) ou `b2w-10k.csv` (pequeno).'\
                                             '\nImportante: Se for utilizar a base de dados menor, ative a flag --comma_separator.',
                        required=True)
    parser.add_argument('-o', type=str, help='Caminho do diretório de destino para os dados preprocessados.'\
                        'Serão salvos os arquivos: `lstm/{tr,te}_reviews.txt`, `lstm/{tr,te}_titles.txt`, '\
                        '`bert/{tr,te}_reviews.txt`, `bert/{tr,te}_titles.txt`, e `bert/{tr,te}_labels.txt`. \n'\
                        'Apesar de separarmos os arquivos do BERT e da LSTM, o split de treino-teste é o mesmo para os dois modelos.',
                        required=True)
    parser.add_argument('--train_size', type=float, help='Proporção do conjunto de dados de treinamento em relação à base toda. Por padrão, 0.7',
                        default=0.8)
    parser.add_argument('--bert_max_review_len', type=int, default=58,
                        help='Número máximo de tokens para um review. Será usado para preparar as bases do BERT.\n'\
                             'A truncagem é feita com base no tokenizador do BERT pré-treinado, e portanto não pode ser utilizada para a LSTM.\n'\
                             'Caso deseje controlar a truncagem para a LSTM, existe um parâmetro para isso no script de treinamento `train_lstm.py`.\n '\
                             'Para entender porque o truncamento é feito nesta etapa para o BERT, consulte README.md. '\
                             'Por padrão, utiliza o tamanho máximo de 58 tokens, que é próximo do quantil 0.8 da distribuição de tamanhos de reviews.')
    parser.add_argument('--bert_max_title_len', type=int, default=12,
                        help='Analogamente ao parâmetro `--bert_max_review_len`, controla o número máximo de tokens de um título.\n'\
                             'Por padrão, são utilizados 12 tokens, que equivale ao quantil 0.8 da distribuição de tamanhos de títulos.')
    parser.add_argument('--seed', type=int, default=42, help='Semente para aleatorização. [42]')
    parser.add_argument('--comma_separator', action='store_true', help='Usa vírgula como separador de campos no arquivo csv. Necessário para a base pequena da b2W.')
    args = parser.parse_args()
    
    print('Iniciando carregamento dos dados...')
    data = load_data(args.i, comma_separator=args.comma_separator)
    print('Dados carregados com sucesso')

    print(f'Iniciando partilha dos dados (random_seed={args.seed})...')
    data_tuple = split_data(data, args.train_size, args.seed)

    save_lstm_data(args.o, *data_tuple)

    print(f'Os arquivos de dados foram salvos em {args.o}/lstm com sucesso.')

    prepare_and_save_bert_seqs(args.o, *data_tuple, bert_max_review_len=args.bert_max_review_len,
                               bert_max_title_len=args.bert_max_title_len)

    print(f'Os arquivos de dados foram salvos em {args.o}/bert com sucesso.')