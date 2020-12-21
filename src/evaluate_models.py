import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score

parser = argparse.ArgumentParser('Cálcula métricas de performance BLEU e METEOR.')
parser.add_argument('--max_ngrams', type=int, help='Maior número de n-gramas para cálculo do BLEU. Serão calculados BLEU-1, ..., BLEU-`max_ngrams`. Por padrão, 3.',
                    default=4)

pred_path = Path(__file__).parent.parent / 'predictions'
p_data_path = Path(__file__).parent.parent / 'prepared_data/lstm/'

stemmer = SnowballStemmer('portuguese')

def cumulative_bleu_scores(titles, predictions, ngram_order=4):
    results = np.zeros(ngram_order)
    
    # Obtém título e predição para o texto `text_nb`.
    titles = [[title] for title in titles]
        
    for ngram in range(ngram_order):
        # Obtém o simplex de dimensão `ngram` e concatena o restante do vetor com zeros
        # p/ que este tenha dimensão 4.
        weighting = tuple(1/(ngram+1) if i <= ngram else 0 for i in range(ngram_order))
        results[ngram] = corpus_bleu(titles, predictions, weights=weighting,
                                              smoothing_function=SmoothingFunction().method2)

    
    return results

if __name__ == '__main__':
    args = parser.parse_args()

    assert (p_data_path / 'te_reviews.txt').exists(), 'Arquivo prepared_data/te_reviews.txt não encontrado.'
    assert (p_data_path / 'te_titles.txt').exists(), 'Arquivo prepared_data/te_titles.txt não encontrado.'

    print('Carregandos dados de teste em prepared_data/lstm/te_* ...')
    titles = [title.strip().lower() 
              for title in open(p_data_path / 'te_titles.txt', encoding='utf-8').readlines()]

    # Obtém a lista de modelos com previsões salvas em predictions/*
    model_list = [model for model in ['lstm', 'bert_cls', 'bert_mask']
                  if (pred_path / f'{model}.txt').exists()]
    
    print(f'Foram encontradas previsões para os modelos: {model_list}')
    print('Carregandos previsões...')

    model_predictions = {
       model: [line.strip().lower() for line in open(pred_path / f'{model}.txt').readlines()]
               for model in model_list
    }

    # Tokeniza labels e predições para calcular BLEU e ACC
    tokenized_titles = [word_tokenize(title, language='portuguese') for title in titles]
    tokenized_model_predictions = {
        model: [word_tokenize(pred, language='portuguese') for pred in predictions]
        for model, predictions in model_predictions.items()
    }

    print(f'Calculando métricas...')

    # METEOR
    avg_meteor_score = {
        model: [np.mean([single_meteor_score(titles[i], pred, stemmer=stemmer)
                        for i, pred in enumerate(predictions)])]
        for model, predictions in model_predictions.items()
    }
    meteor_scores = pd.DataFrame(avg_meteor_score).T
    meteor_scores.columns = ['METEOR']

    # BLEU
    bleu_scores = {
        model: cumulative_bleu_scores(tokenized_titles, predictions, ngram_order=args.max_ngrams)
        for model, predictions in tokenized_model_predictions.items()
    }
    bleu_scores = pd.DataFrame(bleu_scores).T
    bleu_scores.columns = [f'BLEU-{i+1}' for i in range(args.max_ngrams)]

    # ACC
    acc_scores = {
        model: [np.mean(np.array(tokenized_titles) == np.array(predictions))]
        for model, predictions in tokenized_model_predictions.items()
    }
    acc_scores = pd.DataFrame(acc_scores).T
    acc_scores.columns = ['ACC']

    metrics = pd.concat([acc_scores, bleu_scores, meteor_scores], axis=1)
    print(metrics)

    metrics.to_csv(str(Path(__file__).parent.parent / 'scores/results.csv'), index=True, sep='\t')
