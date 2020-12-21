# EP2
Autor: Piero Conti Kauffmann (8940810)

----------------------------

## Dependências necessárias

Para executar os códigos deste trabalho, são necessárias as seguintes dependências:

* tensorflow>=2.2 
* nltk>=3.5 
    * tokenizador de sentenças PUNKT pt-br
    * SnowballStemmer
* Arquivo de dados B2W Completo (recomendado) ou amostra
* transformers==3.5.1
* tqdm

## Instruções para treinar os modelos

### Preparar os dados
No diretório raiz da discplina, executar:

```
python3 src/prepare.py -i ./data/raw/b2w-10k.csv -o ./data/prepared
```

Use a flag `--help` para ver todas as opções de processamento.

### Treinar modelo
Garantindo que o arquivo de embeddings pré-treinados está em `./data/embeddings/word2vec_200k.txt`, executar
no diretório raiz da discplina:

```
# Modelos bidirecionais
python3 src/train.py --epochs 100 --dropout-rate=0 --freeze-emb --maxlen 400
python3 src/train.py --epochs 100 --dropout-rate=0.25 --freeze-emb --maxlen 400
python3 src/train.py --epochs 100 --dropout-rate=0.5 --freeze-emb --maxlen 400

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


