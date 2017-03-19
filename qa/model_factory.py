#-*-coding:utf-8-*-
from keras.layers import LSTM, Dense, Activation, Dropout, Input, merge, RepeatVector, Merge, Lambda ,Flatten
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential,Model
from keras.optimizers import RMSprop
from keras.layers import Embedding
from keras import backend as K
from keras.layers.wrappers import TimeDistributed, Bidirectional
import numpy as np

class ModelFactory(object):
    @staticmethod
    def get_similarity(similarity):
        params = {
            'gamma': 1,
            'c' : 1,
            'd' : 2,
        }

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(
            K.sum(K.square(a - b), axis=1, keepdims=True))

        if similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / \
                (1 + K.exp(-1 * params['gamma']
                           * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / \
                (1 + K.exp(-1 * params['gamma']
                           * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    @staticmethod
    def get_lstm_model(model_param, embedding_file, vocab_size):
        hidden_dim = 128
        weights = np.load(embedding_file)
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_good_base')
        QaEmbedding = Embedding(input_dim=vocab_size,
                                output_dim=weights.shape[1],
                                mask_zero=True,
                                # dropout=0.2,
                                weights=[weights])
        question_embedding =  QaEmbedding(question)
        answer_embedding =  QaEmbedding(answer)
        QaQueLstm = Bidirectional(
            LSTM(output_dim=hidden_dim, return_sequences=False))
        QaAnsLstm = Bidirectional(
            LSTM(output_dim=hidden_dim, return_sequences=False))
        QaQueDense = Dense(hidden_dim, activation='tanh')
        QaAnsDense = Dense(hidden_dim, activation='tanh')
        question_enc_1 = QaQueLstm(question_embedding)
        answer_enc_1 = QaAnsLstm(answer_embedding)
        QaDropout = Dropout(0.2)

        question_enc_2 = QaDropout(question_enc_1)
        answer_enc_2  = QaDropout(answer_enc_1)
        question_enc_3 = QaQueDense(question_enc_2)
        answer_enc_3 = QaAnsDense(answer_enc_2)

        similarity = ModelFactory.get_similarity("cosine")
        basic_model = merge([question_enc_3, answer_enc_3],
                            mode=similarity, output_shape=lambda _: (None, 1))
        lstm_model = Model(
            input=[question, answer], output=basic_model, name='basic_model')
        return lstm_model



    @staticmethod
    def get_lstm_share_model(model_param, embedding_file, vocab_size):
        hidden_dim = 200
        weights = np.load(embedding_file)
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_good_base')
        QaEmbedding = Embedding(input_dim=vocab_size,
                                output_dim=weights.shape[1],
                                # dropout=0.2,
                                weights=[weights])
        question_embedding =  QaEmbedding(question)
        answer_embedding =  QaEmbedding(answer)
        QaQueLstm = Bidirectional(
            LSTM(output_dim=hidden_dim, return_sequences=True))

        QaFlat = Flatten()

        QaQueDense = Dense(hidden_dim, activation='tanh')
        question_enc_1 = QaFlat(QaQueLstm(question_embedding))
        answer_enc_1 = QaFlat(QaQueLstm(answer_embedding))

        QaDropout = Dropout(0.2)

        question_enc_2 = QaDropout(question_enc_1)
        answer_enc_2  = QaDropout(answer_enc_1)
        question_enc_3 = QaQueDense(question_enc_2)
        answer_enc_3 = QaQueDense(answer_enc_2)

        similarity = ModelFactory.get_similarity("cosine")
        basic_model = merge([question_enc_3, answer_enc_3],
                            mode=similarity, output_shape=lambda _: (None, 1))
        lstm_model = Model(
            input=[question, answer], output=basic_model, name='basic_model')
        return lstm_model

    @staticmethod
    def get_convolutional_lstm_model(model_param, embedding_file, vocab_size):
        hidden_dim = 200
        weights = np.load(embedding_file)
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_good_base')
        QaEmbedding = Embedding(input_dim=vocab_size,
                                output_dim=weights.shape[1],
                                # dropout=0.2,
                                weights=[weights])
        question_embedding =  QaEmbedding(question)
        answer_embedding =  QaEmbedding(answer)
        f_rnn = LSTM(hidden_dim, return_sequences=True)
        b_rnn = LSTM(hidden_dim, return_sequences=True)

        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)

        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)

        cnns = [Convolution1D(filter_length=filter_length,
                          nb_filter=500,
                          activation='tanh',
                          border_mode='same') for filter_length in [1, 2, 3, 5]]


        question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')
        
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True

        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        similarity = ModelFactory.get_similarity("cosine")
        basic_model = merge([question_pool, answer_pool],
                            mode=similarity, output_shape=lambda _: (None, 1))
        lstm_convolution_model = Model(
            input=[question, answer], output=basic_model, name='basic_model')
        return lstm_convolution_model


    @staticmethod
    def get_qa_model(model_param, embedding_file, vovab_size):
        margin = model_param.margin
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='int32', name='question_base')
        answer_good = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_bad_base')

        basic_model = ModelFactory.get_lstm_model(
            model_param, embedding_file, vovab_size)
        good_similarity = basic_model([question, answer_good])
        bad_similarity = basic_model([question, answer_bad])

        loss = merge([good_similarity, bad_similarity],
                     mode=lambda x: K.relu(margin - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        prediction_model = Model(
            input=[question, answer_good], output=good_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        training_model = Model(
            input=[question, answer_good, answer_bad], output=loss, name='training_model')
        training_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        return training_model, prediction_model


    @staticmethod
    def get_qa_share_model(model_param, embedding_file, vovab_size):
        margin = model_param.margin
        question = Input(
            shape=(model_param.enc_timesteps,), dtype='int32', name='question_base')
        answer_good = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(
            shape=(model_param.dec_timesteps,), dtype='int32', name='answer_bad_base')

        basic_model = ModelFactory.get_convolutional_lstm_model(
            model_param, embedding_file, vovab_size)
        good_similarity = basic_model([question, answer_good])
        bad_similarity = basic_model([question, answer_bad])

        loss = merge([good_similarity, bad_similarity],
                     mode=lambda x: K.relu(margin - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        prediction_model = Model(
            input=[question, answer_good], output=good_similarity, name='prediction_model')
        prediction_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        training_model = Model(
            input=[question, answer_good, answer_bad], output=loss, name='training_model')
        training_model.compile(
            loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        return training_model, prediction_model