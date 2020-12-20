import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer, TFBertForMaskedLM, TFBertForSequenceClassification
)


def bert_write(review_texts, tokenizer, bert, max_review_len=58, max_title_len=12):
    """Obtém títulos preditos com o BERT para uma lista de reviews.

    Args:
        review_texts ([str]): Lista com reviews
        tokenizer: Tokenizer associado ao modelo
        bert ([type]): 
        max_review_len (int, optional): [description]. Defaults to 58.
        max_title_len (int, optional): [description]. Defaults to 12.

    Yields:
        [type]: [description]
    """    
    assert isinstance(bert, (TFBertForMaskedLM, TFBertForSequenceClassification))
    
    # Detecta se o modelo passado foi treinado com a task de MLM ou Classificação
    prediction_mode = 'mask' if isinstance(bert, TFBertForMaskedLM) else 'cls'

    # Obtém indíces dos tokens especiais e cria array para os títulos gerados recursivamente
    _CLS, _MASK, _EOS, _SEP  = tokenizer.encode('[MASK] [unused1]')
    whole_titles = ['[MASK]']*len(review_texts)

    # Obtém o tamanho (em tokens) do maior review
    max_len = max(len(review) for review in tokenizer(review_texts, add_special_tokens=False)['input_ids'])
    max_len = min(max_len, max_review_len)

    for _ in range(max_title_len):
        tokenized_input = tokenizer(review_texts, whole_titles, return_tensors='tf', padding='max_length',
                                     max_length=max_len + max_title_len + 4, truncation=True)       
        if prediction_mode == 'mask':
            # Obtém o logit do token mascarado
            logits = bert(tokenized_input)[0]
            logits = logits[tokenized_input['input_ids'] == _MASK]
        else:
            # Obtém o logit do token [CLS]
            logits = bert(tokenized_input)[0]
        
        best_tokens = logits.numpy().argmax(axis=-1)

        tokenized_input_ids = tokenized_input['input_ids'].numpy()
        
        for text_nb, input_ids in enumerate(tokenized_input_ids):
            tokenized_input_ids[text_nb][input_ids == _MASK] = best_tokens[text_nb]
        
        # Remove os tokens antes do primeiro [SEP]
        tokenized_input_ids = [
            input_ids[input_ids.index(_SEP)+1:] for input_ids in tokenized_input_ids.tolist()
        ]
        
        # Remove os tokens depois do último [SEP]
        tokenized_input_ids = [
            input_ids[:input_ids.index(_SEP)] for input_ids in tokenized_input_ids
        ]
        
        # Transforma os índices em texto e adiciona [MASK] no final
        whole_titles = [tokenizer.decode(input_ids) + ' [MASK]'
                        for input_ids in tokenized_input_ids]
    
    for whole_title in whole_titles:
        end_token = whole_title.find('[unused1]')
        whole_title = whole_title[:end_token] if end_token >= 0 else whole_title
        yield whole_title.replace('[MASK]', '').strip()

        
# Define parser do script
parser = argparse.ArgumentParser('Gera predições para os modelos BERT, fine-tunados pelos scripts train_bert_{cls,mask}.py')
parser.add_argument('--prediction_batch_size', type=int, default=30, help='Tamanho do batch usado na etapa de previsão.')
parser.add_argument('--max_title_len', type=int, default=12, help='Número de tokens máximos para um título. '\
                    'Por padrão, utiliza 12 tokens (que corresponde ao quantil 0.9 da distribuição dos tamanhos)')
parser.add_argument('--max_review_len', type=int, default=58, help='Número de tokens máximos para um review. '\
                    'Por padrão, utiliza 58 tokens (que corresponde ao quantil 0.9 da distribuição dos tamanhos)')
parser.add_argument('--only_cls', action='store_true', help='Faz predições apenas para o modelo bert_cls')
parser.add_argument('--only_mask', action='store_true', help='Faz predições apenas para o modelo bert_mask')

if __name__ == "__main__":
    args = parser.parse_args()

    # Obtém caminho da pasta raiz do projeto
    root_dir = Path(__file__).absolute().parent.parent

    # Carrega tokenizador
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[unused1]']
    })

    # Caminhos dos modelos
    cls_path = root_dir / 'models/ft_bert_cls'
    mask_path = root_dir / 'models/ft_bert_mask'

    # Carrega dados de validação (vamos pegar os reviews completos, que preparamos para a lstm)
    te_reviews = [line.rstrip() for line in open(root_dir / 'prepared_data/lstm/te_reviews.txt', encoding='utf-8')]
    model_types = []

    if not args.only_cls:
        model_types += [TFBertForMaskedLM]
        assert mask_path.is_dir()

    if not args.only_mask:
        model_types += [TFBertForSequenceClassification]
        assert cls_path.is_dir()

    output_base_path = root_dir / 'predictions'
    assert output_base_path.is_dir(), f'A pasta {output_base_path} não existe.'

    for model_type in model_types:
        # Obtém caminho para o modelo treinado e o caminho de saída
        model_path = mask_path if isinstance(model_type, TFBertForMaskedLM) else cls_path
        output_file = output_base_path / ('bert_mask.txt' if isinstance(model_type, TFBertForMaskedLM) else 'bert_cls.txt')
        
        # Carrega modelo
        bert = model_type.from_pretrained(model_path)
        print(f'Iniciando previsão para o modelo {type(bert).__name__}...')

        # Define batches para a previsão        
        batch_size = args.prediction_batch_size
        batches = int(len(te_reviews)/batch_size) + 1
        titles = []
        for batch_nb in tqdm(range(batches)):
            try:
                batch = te_reviews[batch_nb*batch_size:(batch_nb+1)*batch_size]
                titles += list(bert_write(batch, tokenizer, bert, max_title_len=args.max_title_len))
            except Exception as e:
                print(f'Erro no batch {batch_nb}: {str(e)}')
                import ipdb; ipdb.set_trace()
        
        titles = [title + '\n' for title in titles]

        # Salva resultados
        open(output_file, 'w', encoding='utf-8').writelines(titles)
        print(f'Arquivo {output_file} salvo com sucesso.')
        
        # Deleta modelo carregado
        del(bert)