# Bert for Seq2Seq customer review summarization

**English description**

*This repository contains my submission to the third assigment given in the Computational Linguistics graduate course in University Of São Paulo, 2020 (MAC-5725).*
   
*The goal of this project is to evaluate the performance of different methods of adapting bidirectional masked-language models for text generation-based tasks, which usually use unidirectional/causal LMs.*
   
*This project implements the following strategies:*
 - BERT-CLS: BERT with a multiclass classifier trained w/ a next-token prediction objective.
 - BERT-MASK: Fine-tuning the original MLM classifier layer, masking the last token in the generated sentence (effectively trying to convert a bidirectional model to a causal model).
 - BiLSTM + Attention Benchmark

*BLEU-n and METEOR scores of the above strategies are compared for a non-extractive text summarization task using the [B2W customer reviews dataset](https://github.com/b2wdigital/b2w-reviews01).* *The final report (written in portuguese) is available [here](https://github.com/piero2c/BertTextGeneration/blob/master/report/main.pdf)*.

*For the specific implementation details, see [`src/`](https://github.com/piero2c/BertTextGeneration/tree/master/src)*.

----------------------------

## Dependências necessárias

Para executar os códigos deste trabalho, são necessárias as seguintes dependências:

* tensorflow==2.3.1
* nltk>=3.5 
    * tokenizador de sentenças PUNKT pt-br
    * SnowballStemmer (necessário para usar a métrica METEOR corretamente)
* Arquivo de dados B2W Completo (recomendado) ou amostra de dez mil linhas
* transformers==3.5.1
* tqdm

## Binários dos modelos (BERT-CLS e BERT-MASK) treinados

Para os modelos BERT-CLS e BERT-MASK, que exigem um esforço computacional maior, disponibilizei os binários dos modelos treinados
no `Google Drive` para consulta. Caso deseje baixar os modelos, use estes links:

* [BERT-CLS](https://drive.google.com/file/d/178WRQ-l5vPb4crORyYqGcllrn4DRDS0z/view?usp=sharing)
* [BERT-MASK](https://drive.google.com/file/d/1uSaoRLbWq-cdcHAdrVHKhu6uxSimPs46/view?usp=sharing)

```
cd models

# BERT-CLS
tar -xzf ft_bert_cls.tar.gz

# BERT-MASK
tar -xzf ft_bert_mask.tar.gz
```

Com os binários na pasta `models`, não é necessário executar os scripts de treinamento `train_bert_cls.py` e `train_bert_mask.py`.

## Títulos gerados pelos modelos treinados e métrica de avaliação humana

Neste repositório, os títulos gerados pelos modelos finais treinados já estão na pasta `predictions` para consulta, assim como as métricas finais em `scores/results.csv`. 
A métrica de avaliação humana usa os primeiros 200 dados da amostra aleatória de testes foi utilizada. Você pode consultar estes dados [neste link](https://docs.google.com/spreadsheets/d/1v9te15-LVNhdp3a1Iksk0YgP8a0gX4CPlsu6Jc0_5cI/edit?usp=sharing).

As métricas ACC, BLEU, METEOR dos modelos finais também podem ser consultadas em `scores/results.csv` caso seja necessário.
Para reproduzir o pipeline completo de treinamento, siga as instruições do guia à seguir.

## Instruções para reproduzir os experimentos

## Baixar os dados
Nos experimentos, utilizei a base de dados completa de reviews da [B2W](https://github.com/b2wdigital/b2w-reviews01/blob/master/B2W-Reviews01.csv), mas também é possível utilizar a versão reduzida de apenas 10 mil linhas. Para isso, basta executar na raiz do diretório do projeto:

```
# Base completa
wget https://github.com/b2wdigital/b2w-reviews01/blob/master/B2W-Reviews01.csv ./data

# Base reduzida
wget https://github.com/alan-barzilay/NLPortugues/blob/master/Semana%2003/data/b2w-10k.csv ./data
```

### Divisão e preparação dos dados para treinamento
No diretório raiz da discplina, executar:

```
python3 src/prepare_data.py -i ./data/B2W-Reviews01.csv -o ./prepared_data
```

**Importante**: Caso você esteja usando a versão reduzida da base da B2W, ative a opção `--comma_separator`.

Use a flag `--help` para ver todas as opções de processamento.

Esse script irá gerar dois diretórios `prepared_data/lstm` e `prepared_data/bert` com os arquivos separados nos conjuntos
de treino e teste. 

Serão criados no diretório `prepared_data/lstm`:
* (tr|te)_reviews.txt: reviews do conjunto de treino/teste
* (tr|te)_titles.txt: títulos do conjunto de treino/teste

E no diretório `prepared_data/bert`:
* (tr|te)_reviews.txt: reviews do conjunto de treino/teste. Note que o mesmo review aparece mais de uma vez, 
isso é necessário devido a divisão da task de geração de texto em multiplas tasks de classificação.
* (tr|te)_titles.txt: títulos mascarados do conjunto de treino/teste seguindo o procedimento descrito no enunciado da tarefa.
* (tr|te)_labels.txt: índices dos tokens (em relação ao tokenizador da NeuralMind) para os tokens mascarados do conjunto de treino/teste.

Apesar dos arquivos gerados serem distintos por conta do preprocessamento ser diferente, é importante notar que o split de treino e teste é o mesmo tanto para a LSTM como para o BERT.

### Treinar a BiLSTM Encoder-Decoder

Para treinar a BiLSTM Encoder-Decoder com mecanismo de atenção, basta executar o script `src/train_and_predict_lstm.py`

```
python3 src/train_and_predict_lstm.py
```

Após o final do treinamento, os pesos da rede treinada serão salvos em `models/lstm_weights` e os títulos gerados para o conjunto de testes serão salvos em `predictions/lstm.txt`.

Use a flag `--help` para ver todas as opções do script, e se atente aos parâmetros `--batch_size`, `--vocab_size`, `--emb_size` e `--prediction_nb_batches` caso queira diminuir o uso de VRAM. 

**Importante**: Na etapa de previsão do conjunto de testes é necessário dividir o dataset em partes para não exceder o limite de VRAM disponível. O parâmetro `--prediction_nb_batches` controla o número de batches para a etapa de previsão e não afeta o modelo treinado. Caso você tenha problemas de falta de memória na etapa de previsão, tente aumentar o valor desse parâmetro (por padrão, 20). 

Se quiser reproduzir o modelo treinado no relatório, utilize os parâmetros default do script. 


### Modelo BERT-CLS

Para treinar o modelo BERT-CLS, utilize o script `src/train_bert_cls.py`.

```
python3 src/train_bert_cls.py
```

O modelo finalizado será salvo na pasta `models/ft_bert_cls`.

Algumas observações importante:
* Caso você tenha alterado os valores das opções `--bert_max_review_len` e `--bert_max_title_len` no script `src/prepare_data.py`, é necessário fornecer os mesmo valores utilizados neste script também, por meio dos parâmetros `--max_review_len` e `--max_title_len`.
* A base de dados da B2W completa é grande e, visto que o preprocessamento dos dados para este modelo aumenta o número de instâncias, pode ser necessário reduzir o número de instâncias de treinamento dependendo de quanta RAM estiver disponível na sua máquina. Para controlar o tamanho do dataset que será utilizado no treinamento, use a opção `--nb_instances`, caso contrário, o dataset inteiro será usado.
* Caso queira diminuir também o uso de VRAM, se atente às opções `--nb_grad_acc` e `--batch_size`.

Use a flag `--help` para ver todas as opções do script.

### Modelo BERT-MASK

O treinamento do BERT-MASK é mais custoso computacionalmente pois exige tamanhos de *batchs* maiores. Recomendo utilizar a infra-estrutura do `Google Colab`. Assim como o BERT-CLS, para treinar o modelo basta usar o script `src/train_bert_mask.py`

```
python3 src/train_bert_mask.py
```

O modelo finalizado será salvo na pasta `models/ft_bert_mask`.

As mesmas observações do modelo BERT-CLS também se aplicam aqui:
* Caso você tenha alterado os valores das opções `--bert_max_review_len` e `--bert_max_title_len` no script `src/prepare_data.py`, é necessário fornecer os mesmo valores utilizados neste script também, por meio dos parâmetros `--max_review_len` e `--max_title_len`.
* A base de dados da B2W completa é grande e, visto que o preprocessamento dos dados para este modelo aumenta o número de instâncias, pode ser necessário reduzir o número de instâncias de treinamento dependendo de quanta RAM estiver disponível na sua máquina. Para controlar o tamanho do dataset que será utilizado no treinamento, use a opção `--nb_instances`, caso contrário, o dataset inteiro será usado.
* Caso queira diminuir também o uso de VRAM, se atente às opções `--nb_grad_acc` e `--batch_size`.

Use a flag `--help` para ver todas as opções do script.

### Realizar predição para os modelos BERT-CLS e BERT-MASK

Com os binários dos modelos treinados na pasta `models/ft_bert_cls` e `models/ft_bert_mask`, é possível executar o script de geração dos títulos usando o script `src/predict_bert.py`.

```
python3 src/predict_bert.py
```

As previsões serão salvas em `predictions/bert_cls.txt` e `predictions/bert_mask.txt`. (Este repositório já inclui estes arquivos caso você queira consulta-los)

Caso deseje fazer a previsão apenas do modelo BERT-CLS ou do modelo BERT-MASK, use as flags `--only_cls` ou `--only_mask`. Por padrão, será calculada a previsão para os dois modelos.

Repito aqui algumas observações importantes:
* Caso você tenha alterado os valores das opções `--bert_max_review_len` e `--bert_max_title_len` no script `src/prepare_data.py`, é necessário fornecer os mesmo valores utilizados neste script também, por meio dos parâmetros `--max_review_len` e `--max_title_len`.
* Caso queira diminuir o uso de VRAM, se atente às opções `--prediction_batch_size`


### Avaliar performance de todos os modelos

Com todas as previsões salvas na pasta `predictions`, as métricas podem ser obtidas usando o script `src/evaluate_models.py`


```
python3 src/evaluate_models.py
```

Consulte as opções usando a flag `--help`.
O arquivo de saída produzido com as métricas será salvo em `scores/results.csv` (caso queira consultar, este arquivo também já está salvo).

