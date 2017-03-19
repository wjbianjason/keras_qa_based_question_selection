#-*-coding:utf-8-*-
import copy
import numpy as np
import logging
import sys
from scipy.stats import rankdata
import os

log = logging.getLogger("output")
logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='log.txt',
                filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
global_predict_epoch = 0


def save_epoch(model, epoch, prefix):
    if not os.path.exists('model/'):
        os.makedirs('model/')
    model.save_weights(
        'model/'+prefix+'weights_epoch_%d.h5' % epoch, overwrite=True)


def load_epoch(model, epoch, prefix):
    assert os.path.exists('model/'+prefix+'weights_epoch_%d.h5' %
                          epoch), 'Weights at epoch %d not found' % epoch
    model.load_weights('model/'+prefix+'weights_epoch_%d.h5' % epoch)


def count_corpus_num(generator, article_filename, title_filename):
    a = generator.iterFilterSummaryData(
        article_filename, title_filename, count_flag=True)
    while True:
        try:
            a.next()
        except Exception as e:
            print e
            break
    return generator.corpus_amount


def prog_bar(so_far, total, n_bars=20):
    n_complete = int(so_far * n_bars / total)
    if n_complete >= n_bars - 1:
        sys.stderr.write('\r[' + '=' * n_bars + ']')
    else:
        s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * \
            (n_bars - n_complete) + ']'
        sys.stderr.write(s)


def main(mode="train"):
    from qa.model_factory import ModelFactory
    from qa.data_process import Vocab, DataGenerator, ModelParam
    model_param = ModelParam(enc_timesteps=150, dec_timesteps=150, batch_size=64, random_size=5, margin=0.05)
    # total num = 22352
    vocab_all = Vocab("./data/vocab_all.txt", max_size=80000)
    data_generator = DataGenerator(
        vocab_all, model_param, answer_file="./data/answers.pkl")
    embedding_file = "./data/word2vec_100_dim.embeddings"
    train_model, predict_model = ModelFactory.get_qa_share_model(
        model_param, embedding_file, vocab_all.NumIds())
    train_model.summary()
    if mode == "train":
        epoch = 120
        train_filename = "./data/train.pkl"
        val_loss = {'loss': 1., 'epoch': 0}
        for i in range(epoch):
            questions, good_answers, bad_answers = data_generator.trainDataGenerate(
                train_filename)
            logging.info('Fitting epoch %d' % i)
            label_no_use = np.zeros(shape=(questions.shape[0],)) # doesn't get used
            hist = train_model.fit([questions, good_answers, bad_answers],label_no_use,
                                   nb_epoch=1, batch_size=model_param.batch_size, validation_split=0.1, verbose=1)
            logging.info('Epoch %d ' % i + 'Loss = %.4f, Validation Loss = %.4f ' % (hist.history['loss'][0], hist.history['val_loss'][0]) + '(Best: Loss = %.4f, Epoch = %d)' % (val_loss['loss'], val_loss['epoch']))    
            save_epoch(train_model,i, prefix="insurance_train")
            save_epoch(predict_model,i, prefix="insurance_predict")
            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}                

    elif mode == "evaluate":
    	verbose = False # print acc in detail
        eva_list = ["./data/dev.pkl", "./data/test1.pkl", "./data/test2.pkl"]
        eva_filename = "./data/dev.pkl"
        load_epoch(predict_model, global_predict_epoch, "insurance_predict")
        data = data_generator.evaluateDataGenerate(eva_filename)
        c_1, c_2 = 0, 0
        logging.info("loading done,evaluating...")
        for i, d in enumerate(data):
            prog_bar(i, len(data))
            indices, answers, question = data_generator.processData(d)
            sims = predict_model.predict([question, answers])

            n_good = len(d['good'])
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])

            r = rankdata(sims, method='max')

            if verbose:
                min_r = np.argmin(sims)
                amin_r = data_generator.answers[indices[min_r]]
                amax_r = data_generator.answers[indices[max_r]]
                amax_n = data_generator.answers[indices[max_n]]

                logging.info(' '.join(vocab_all.Revert(d['question'])))
                logging.info('Predicted: ({}) '.format(
                    sims[max_r]) + ' '.join(vocab_all.Revert(amax_r)))
                logging.info('Expected: ({}) Rank = {} '.format(
                    sims[max_n], r[max_n]) + ' '.join(vocab_all.Revert(amax_n)))
                logging.info('Worst: ({})'.format(sims[min_r]) + ' '.join(vocab_all.Revert(amin_r)))

            c_1 += 1 if max_r == max_n else 0
            c_2 += 1 / float(r[max_r] - r[max_n] + 1)

        top1 = c_1 / float(len(data))
        mrr = c_2 / float(len(data))

        del data
        logging.info('Top-1 Precision: %f' % top1)
        logging.info('MRR: %f' % mrr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.error("need at least one param")
        exit(1)
    flag = sys.argv[1]
    if flag == "evaluate":
        try: 
            global_predict_epoch = int(sys.argv[2])
            main(flag)
        except Exception as e:
            # print e
            logging.error("the second param should be number")
    elif flag == "train":
        main(flag)
    else:
        logging.error("the first param should be train or evaluate")
