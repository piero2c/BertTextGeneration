from time import sleep
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Encoder(keras.Model):
    def __init__(self, lstm_hidden=50):
        super(Encoder, self).__init__()
        self.lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_hidden, return_sequences=True, return_state=True)
        )

    def call(self, tokenized_text):
        # Converte tokens em embeddings
        # tokenized_text = self.embedding_layer(tokenized_text)
        
        # Passa os embeddings p/ uma camada LSTM 
        seq_hidden_states, left_h, left_c, right_h, right_c = self.lstm(tokenized_text)
        
        last_h = tf.concat([left_h, right_h], axis=1)
        last_c = tf.concat([left_c, right_c], axis=1)
        
        # Retorna a sequência de unidades ocultas e o último estado da lstm        
        return seq_hidden_states, [last_h, last_c]

class TeacherForcedDecoder(keras.Model):
    def __init__(self, vocab_size, lstm_hidden=100):
        super(TeacherForcedDecoder, self).__init__()
                
        # LSTM pós camada de embeddings
        self.lstm = keras.layers.LSTM(lstm_hidden, return_sequences=True)
        
        # LSTM pós camada de atenção
        self.context_augmented_lstm = keras.layers.LSTM(lstm_hidden, return_sequences=True)

        # Camada de dropout
        self.drop = keras.layers.Dropout(0.5)
        
        # Camada final de saída do decoder
        self.fc1 = keras.layers.Dense(300, activation='selu')
        self.output_layer = keras.layers.Dense(vocab_size)

    def calc_attention(self, encoder_hidden, decoder_hidden):
        '''
            Inputs:
                encoder_hidden (batch_size, encoder_length, hidden)
                decoder_hidden (batch_size, decoder_length, hidden)
            Output:
                (
                    context_vector
                )
        '''
        # Obtem attention_score (batch_size, encoder_length, decoder_length)
        attention_scores = tf.matmul(encoder_hidden, decoder_hidden, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)
        
        # Calcula o vetor de contextos como uma média ponderada de `encoder_hidden` 
        # segundo os escores de atenção.
        context_vectors = tf.matmul(attention_scores, encoder_hidden, transpose_a=True)
        
        # Retorna
        #   - Os vetores de contexto (batch_size, decoder_length, hidden)
        #   - Os escores de atenção (batch_size, encoder_length, decoder_length)
        return context_vectors, attention_scores
        
    def call(self, ouput_emb, encoder_h, encoder_end_state, training=False):
        '''
            Recebe a sequência de saída completa, por exemplo:
                [<s>, Produto, muito, bom, </s>, <pad>, <pad>]
            
            E prevê com o próximo token da sequência de saída, por exemplo:
                [Produto, muito, bom, </s>, <pad>, <pad>, <pad>]
            
            Note que são passados sempre os tokens corretos da sequência como entrada, isto é, se o decodificador
            prever que o próximo token da sequência '<s> Produto' é o token 'ruim', isso será considerado 
            como um erro (sendo assim penalizado na função de perda), mas para t+1 será fornecido o token 'muito'.
            
            Essa metodologia de aprendizado é conhecida como 'teacher forcing', pois forçamos no modelo
            a sequência correta de tokens. Isso é útil pois acelera a convergência de modelos seq2seq e 
            simplifica a implementação do modelo.
            
            Numa situação real de inferência, basta fornecer o token <s> como `output_text` para
            obter uma previsão para o próximo token. Depois, basta retroalimentar a rede
            com o novo token output. Este procedimento deve ser repetido até se obter o token de fim
            de sentença </s>. 
        '''
        # Obtém embeddings da sequência de saída
        # ouput_emb = self.embedding_layer(output_text)
        
        # Obtém os hidden states da sequência de saída (batch_size, decoder_length, hidden)
        # é importante que esta LSTM utilize o último estado da LSTM do Encoder.
        output_h = self.lstm(ouput_emb, initial_state=encoder_end_state)
        
        # Obtém o vetor contexto (batch_size, decoder_length, hidden) a partir dos
        # pesos de atenção
        context_vector, att_weights = self.calc_attention(encoder_h, output_h)
        
        # Concatena o vetor de contextos aos hidden states `output_h`,
        # produzindo o tensor (batch_size, decoder_length, 2*hidden)
        output_h = tf.concat([output_h, context_vector], axis=-1)
        
        # Passa o resultado para uma lstm final
        # Não é obrigatório, mas vamos usar os estados finais da LSTM do encoder também
        final_h = self.context_augmented_lstm(output_h, initial_state=encoder_end_state)
        
        # Aplica dropout em final_h
        if training:
            final_h = self.drop(final_h)
        
        # Transforma o resultado em um tensor (batch_size, decoder_length, vocab_size)
        #output = self.output_layer(final_h)
        output_h = self.fc1(output_h)
        output = self.output_layer(output_h)
        
        # Retorna o output do modelo e os pesos de atenção (para análise)
        return output, att_weights

def build_model(max_len_X, max_len_y, vocab_size, emb_dim):
    # Define inputs
    input_review = keras.Input(shape=max_len_X, dtype="int32")
    input_review_title = keras.Input(shape=max_len_y, dtype="int32")
    
    # Cria encoder e decoder
    encoder = Encoder()
    decoder = TeacherForcedDecoder(vocab_size+1)
    
    # Cria camada de embedding e de masking
    emb_layer = keras.layers.Embedding(vocab_size+1, emb_dim)
    mask_layer = keras.layers.Masking(mask_value=0)
    
    # Aplica mascara nos reviews e converte review e titulo em embeddings
    masked_review_title = mask_layer(input_review_title)
    x = emb_layer(input_review)
    y = emb_layer(input_review_title)
    
    # Aplica encoder e decoder
    encoded_review_h, encoded_review_last_state = encoder(x)
    decoded_title, _ = decoder(y, encoded_review_h, encoded_review_last_state)
    
    # Retorna keras.Model
    return keras.Model(inputs=[input_review, input_review_title], outputs=decoded_title)
  
def write_title(model, text2idx, review_text_idx, return_also_idx=False):
    start = ['<s>']
    start_idx = [text2idx.word_index['<s>']]
    
    for i in range(11):
        out = model.predict([
            np.array([review_text_idx]), pad_y(text2idx.texts_to_sequences([start]))
        ])
        
        next_token = text2idx.sequences_to_texts(
            [[out.argmax(axis=-1)[0][i]]]
        )[0]
        
        start.append(next_token)
        start_idx.append([out.argmax(axis=-1)[0][i]][0])
        
        if next_token == '</s>':
            break
    
    if return_also_idx:
        return start, np.array(start_idx)
    
    return start


def write_titles(model, text2idx, review_texts_idx, return_also_idx=False, max_title_len=11):
    pad_y = lambda y: pad_sequences(y, maxlen=max_title_len, padding='post')

    """Alimenta recursivamente o modelo para gerar textos"""    
    # Inicializa os títulos
    start = [['<s>'] for _ in range(len(review_texts_idx))]
    start_idx = [[text2idx.word_index['<s>']] for _ in range(len(review_texts_idx))]
    
    # Alimenta retroativamente a predição da rede
    for i in range(max_title_len):
        next_tokens_idx = model.predict([
            np.array(review_texts_idx), pad_y(text2idx.texts_to_sequences(start))
        ], batch_size=8).argmax(axis=-1)[:, i, True]

        next_tokens = text2idx.sequences_to_texts(next_tokens_idx)
        
        for text_nb, next_token in enumerate(next_tokens):
            start[text_nb] += [next_token]
            start_idx[text_nb] += next_tokens_idx[text_nb].tolist()

    # Remove tudo após o token </s>
    start = [start_i[0:start_i.index('</s>')+1] if '</s>' in start_i else start_i
             for start_i in start]
    start_idx = [start_idx[i][0:len(start[i])+1] for i in range(len(start))]
    
    if return_also_idx:
        return start, np.array(start_idx)
    
    return start