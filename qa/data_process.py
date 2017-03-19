#-*-coding:utf-8 -*-
# from __future__ import print_function
import sys
import numpy as np
import nltk
import random
from collections import namedtuple
import copy
import pickle
import logging
from scipy.stats import rankdata


logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='log.txt',
                filemode='a')

ModelParam = namedtuple("ModelParam","enc_timesteps,dec_timesteps,batch_size,random_size,margin")

# SENTENCE_START = '<s>'
# SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
class Vocab(object):
  	"""Vocabulary class for mapping words and ids."""
	def __init__(self, vocab_file, max_size):
		self._word_to_id = {}
		self._id_to_word = {}
		self._count = 0
		for word in [PAD_TOKEN]:
				self.CreateWord(word)
		with open(vocab_file, 'r') as vocab_f:
			for line in vocab_f:
				pieces = line.split()
				if len(pieces) != 2:
					sys.stderr.write('Bad line: %s\n' % line)
					continue
				if pieces[1] in self._word_to_id:
					raise ValueError('Duplicated word: %s.' % pieces[1])
				self._word_to_id[pieces[1]] = self._count
				self._id_to_word[self._count] = pieces[1]
				self._count += 1
				if self._count > max_size-1:
					sys.stderr.write('Too many words: >%d.' % max_size)
		          		break

	def WordToId(self, word):
		if word not in self._word_to_id:
			return self._word_to_id[UNKNOWN_TOKEN]
		return self._word_to_id[word]

	def IdToWord(self, word_id):
		if word_id not in self._id_to_word:
			raise ValueError('id not found in vocab: %d.' % word_id)
		return self._id_to_word[word_id]

	def NumIds(self):
		return self._count

	def CreateWord(self,word):
		if word not in self._word_to_id:
			self._word_to_id[word] = self._count
			self._id_to_word[self._count] = word
			self._count += 1
	def Revert(self,indices):
		vocab = self._id_to_word
		return [vocab.get(i, 'X') for i in indices]


class DataGenerator(object):
  	"""Dataset class
 	"""
  	def __init__(self,vocab,model_param,answer_file):
		self.vocab = vocab
		self.param = model_param
		self.batch_size = self.param.batch_size
		self.corpus_amount = 0
		self.answers = pickle.load(open(answer_file,'rb'))

	def padq(self, data):
	    return self.pad(data, self.param.enc_timesteps)

	def pada(self, data):
	    return self.pad(data, self.param.dec_timesteps)

	def pad(self, data, len=None):
	    from keras.preprocessing.sequence import pad_sequences
	    return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

	def trainDataGenerate(self,filename):
		training_set = pickle.load(open(filename,'rb'))
		questions = list()
		good_answers = list()
		indices = list()

		for j, q in enumerate(training_set):
		    questions += [q['question']] * len(q['answers']) # each answer generate a sample
		    good_answers += [self.answers[i] for i in q['answers']] # each answer generate a label
		    indices += [j] * len(q['answers']) # recored every sample's question's index

		logging.info('Began training on %d samples' % len(questions))
		self.corpus_amount = len(questions)        

		questions = self.padq(questions)
		good_answers = self.pada(good_answers)
		bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))
		return questions,good_answers,bad_answers

	def evaluateDataGenerate(self,filename):
		data = pickle.load(open(filename,'r'))
		random.shuffle(data)
		return data
		
	def processData(self,d):
		indices = d['good'] + d['bad']
		answers = self.pada([self.answers[i] for i in indices])
		question = self.padq([d['question']] * len(indices))
		return indices,answers,question

	




class Util(object):
	@staticmethod
	def generate_qa_vocab(output="vocab_all.txt"):
		vf = open("../data/"+output,'w')
		handle = open("../data/vocabulary.pkl",'r')
		vocab_dic = pickle.load(handle)
		print len(vocab_dic)
		for index,word in vocab_dic.items():
			vf.write(str(index)+" "+word+"\n")

if __name__ == "__main__":
	Util.generate_qa_vocab()
