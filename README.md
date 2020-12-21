# EP2
Autor: Piero Conti Kauffmann (8940810)

----------------------------

## Dependências necessárias

Para executar os códigos deste trabalho, são necessárias as seguintes dependências:

* tensorflow==2.3.1
* nltk>=3.5 
    * tokenizador de sentenças PUNKT pt-br
    * SnowballStemmer
* Arquivo de dados B2W Completo (recomendado) ou amostra
* transformers==3.5.1
* tqdm

## Binários dos modelos (BERT-CLS e BERT-MASK) treinados
Para os modelos BERT-CLS e BERT-MASK, que exigem um esforço computacional maior, disponibilizei os binários dos modelos treinados
no `Google Drive`. Você pode inclui-los na raiz do projeto executando:

```
cd models

# BERT-CLS
wget 

# BERT-MASK
```

Com os binários na pasta models, não é necessário executar os scripts de treinamento `train_bert_cls.py` e `train_bert_mask.py`.

## Instruções para reproduzir os experimentos

## Baixar os dados
Nos experimentos, utilizei a base de dados completa de reviews da ![B2W](https://github.com/b2wdigital/b2w-reviews01/blob/master/B2W-Reviews01.csv), mas também é possível utilizar a versão reduzida de apenas 10 mil linhas. Para isso, basta executar na raiz do diretório do projeto:

```
# Base completa
wget https://github.com/b2wdigital/b2w-reviews01/blob/master/B2W-Reviews01.csv ./data

# Base reduzida
wget https://github.com/alan-barzilay/NLPortugues/blob/master/Semana%2003/data/b2w-10k.csv ./data
```

### Divisão e preparação dos dados para treinamento
No diretório raiz da discplina, executar:

```
python3 src/prepare.py -i ./data/B2W-Reviews01.csv -o ./prepared_data --train_size 0.8
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

# Modelos unidirecionais
python3 src/train.py --epochs 100 --unidirectional --dropout-rate=0.0 --freeze-emb --maxlen 400
python3 src/train.py --epochs 100 --unidirectional --dropout-rate=0.25 --freeze-emb --maxlen 400
python3 src/train.py --epochs 100 --unidirectional --dropout-rate=0.5 --freeze-emb --maxlen 400
```
Os modelos escolhidos via `early-stopping` estarão na pasta `./models` assim como os arquivos `training_logs.csv` respectivos com o log de treinamento completo dos modelos.
Use a flag `--help` para ver todas as opções de treinamento.

### Avaliar performance do modelo

No diretório raiz, executar:

```
python3 src/evaluate.py
```

Os gráficos gerados estarão no diretório `./report/`. 
A tabela de resultados finais é apresentada no `stdout` do script.


