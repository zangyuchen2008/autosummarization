import jieba
import numpy as np 
from sklearn.decomposition import PCA
from bert_serving.client import BertClient

#followed essay:  simple but tough to beat baseline for sentense embdeeding, without removing 1st priciple component via pca 
# full code: https://github.com/zangyuchen2008/a_simple_but_tough_to_beat_baseline_for_sentence_embeddings
def sentence_to_vec(sentence_list, embedding_size,model, word_sifs,stop_words, a=1e-3):
    sentence_set = [] #store weighted sentence vec
    sentence_ls = [] #store sentece list which has vec
    for sentence in sentence_list:
        vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
        cuts = list(jieba.cut(sentence))
        cuts = [ w for w in cuts if ((w in model) and (w not in stop_words))] # make sure tokens in word2v model, ignor thoese not
        sentence_length = len(cuts)

        for word in cuts:
            if word in word_sifs: para = word_sifs[word]
            else: para = 1    
            a_value = a / (a + para)  # smooth inverse frequency, SIF
            vs = np.add(vs, np.multiply(a_value, model[word]))  # vs += sif * word_vector

        vs = np.divide(vs, sentence_length)  # weighted average
        if np.isnan(vs).any() == True : #if there is sentece vec equals nan, then skip
            continue
        sentence_set.append(vs)  # add sif weighted sentence vec to a vec list
        sentence_ls.append(sentence) #add raw sentencs that have vec 
    
    return sentence_set , sentence_ls


class bert_sentence_vec:
    def __init__(self,sens,bc=BertClient()):
        super().__init__()
        self.bc = bc
        self.sens = sens
    def get_vec(self):
        return self.bc.encode(self.sens)

if __name__ == "__main__":
    ber_sevice = bert_sentence_vec(['今天天气很好'])
    print(ber_sevice.get_vec())