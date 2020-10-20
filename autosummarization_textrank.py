# coding=utf8
# import gensim
# import pickle
# from gensim.models import Word2Vec,LdaModel,KeyedVectors
# from gensim.models.callbacks import CallbackAny2Vec
# import jieba
# import numpy as np
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cosine
import re
# from collections import Counter
# import math
# from sen2vec import sentence_to_vec
# import os
# import warnings
# import networkx as nx 
# warnings.filterwarnings("ignore")

# split sentence 同时可以恢复标点符号
def split_sentence(sentence):
    result= re.split('[。?？!！\r\n]',sentence)
    senswith_notation=[]
    for index , s in enumerate(result):
        backward_length = list(map(len,result[:index+1]))
        total_length = sum(backward_length) + index #每次分裂都会少一个字符，所以要加上index
        if total_length< len(sentence):
            notation = sentence[total_length]
            senswith_notation.append(s + notation)
    result = list(filter(lambda x: len(x)>3,senswith_notation))
    result_cleaned = []
    while result:
        sen = result.pop()
        if np.sum([str.strip(sen) == str.strip(s) for s in result])==0 and \
            bool(re.search('[。?？!！]',sen)): result_cleaned.insert(0,sen) # remove duplicated sentence, remove subtitle in the document (which has no puctuation)
        else: continue
    return result_cleaned

# get sentence tfidf
def get_tfidfs(sentece,word_idf,stop_words,threshhold):
    words_filtered = list(filter(lambda x :bool(re.findall('\w+',x)) and (x not in stop_words) , jieba.cut(sentece)) )
    sen_word_count = Counter(words_filtered)
    tfidfs = []
    for a, b in sen_word_count.most_common():
        if a in word_idf:
            tfidfs.append((a,  b/len(words_filtered) * word_idf[a]))
        else :
            tfidfs.append((a,  b/len(words_filtered)))
    return sorted(tfidfs,key=lambda x: x[1],reverse=True)[:threshhold]

# get textrank result
def get_textrank(sens_embedding, sens,tfidfs,para_title, para_keyword,para_fisrtsen):
    # keyword overlap
    sens_keywords = {}
    key_words = [a for a,b in tfidfs]
    for index, sen in enumerate(sens):
        words = list(jieba.cut(sen))
        sens_keyword = [w for w in words if w in key_words]
        sens_keywords[index] = sens_keyword
    # create graph
    G= nx.Graph()
    edges = []
    for i , v1 in enumerate(sens_embedding):
        for j, v2 in enumerate(sens_embedding[i+1 :]):
            com_keyword_num = len(set(sens_keywords[i]) &  set(sens_keywords[i+j+1]))
            # we decrease cosin distance between sens based on commmon key word number
            score = cosine(v1,v2)*(1- com_keyword_num*para_keyword) 
            if i ==0:
                score = score * para_title # weight for relation with title 
                edges.append((i,j+i+1,score))
            else:
                edges.append((i,j+i+1,score))
    G.add_weighted_edges_from(edges)
    # pagerank
    page_rank = nx.pagerank(G,weight='weight')
    # weight first sentense
    page_rank[1] = page_rank[1] * para_fisrtsen
    # sorted based on ranking values
    result = sorted(zip(page_rank.items(),sens,sens_embedding),key=lambda x: x[0][1])
    return result, G

def autosummation(title,doc,model,word_sifs,stop_words,word_idf):
    # get titile+document embedding
    spl_sens = split_sentence(title+ '。'+doc)
    sens_embedding1 , spl_sens_cleared = sentence_to_vec(spl_sens,100,model,word_sifs,stop_words,)
    # get document keywords via tfidf
    tfidfs = get_tfidfs(doc,word_idf,stop_words,10)
    # get textrank
    result ,G = get_textrank(sens_embedding1,spl_sens_cleared,tfidfs,0.5,0.05,0.8)
    # sort based on original sequence in document
    key_sentences = sorted(result[:4],key= lambda x: x[0][0])
    return ''.join([b for a,b,c in key_sentences])



if __name__ == '__main__':
    # load model
    # model_path = os.path.join(os.path.abspath('./'),'word2vector_Model','word2vec.kv')
    # model = KeyedVectors.load(model_path,mmap='r')
    # # load sif
    # word_sifs =pickle.load(open('data\word_sifs.plk','rb'))
    # # load stopwords
    # stop_words = pickle.load(open('data\stop_words.plk','rb'))
    # # load idf
    # word_idf = pickle.load(open('data\word_idf.plk','rb'))
    # test
    title= '''中美联合研究：3月1日美国或已有9484例新冠感染'''
    doc = '''
    美国现在有多少新冠肺炎患者？截至美国东部时间3月9日19时，美国有线电视新闻网给出的累计确诊数字为至少704。

然而，一项基于武汉直飞数据的研究评估认为，在3月1日，美国仅从武汉直接输入的病例就造成了1043至9484例感染。



3月6日，一个中美联合研究团队在预印本网站medRxiv上合作发表了题为《评估COVID-19在美流行规模：基于中国武汉直飞航线的模拟》（Estimating the scale of COVID-19 Epidemic in the United States: Simulations Based on Air Traffic Directly from Wuhan, China ）的论文，尚未经过同行评议。

论文的通讯作者来自位于美国洛杉矶的西达赛奈（Cedars-Sinai）医疗中心。北京大学流行病与卫生统计系教授李立明、曹卫华和吕筠参与研究。

研究团队声明，分析有意采取了一系列保守性假设，再加上模型简化，可能会出现偏差，需读者理性判断。

例如，该研究只考虑武汉在封锁（1月23日）前直飞美国的人群，未计算从中国其余地区或其他国家（如韩国、意大利或伊朗）输入的病例。美国在3月1日前采取的相关监测和隔离措施也考虑进去了。

此外，研究假设这些武汉输入的病例在美诊断后就停止传播病毒。

武汉天河机场目前有两条直飞美国的航线，目的地分别是旧金山和纽约，鉴于相应期限的航空数据尚未更新，研究参考了以往可类比的数据。

截至论文写作的2月29日，美国公开报告了20个病例的信息，其中8例在发病前到过武汉，1例有过北京旅行史，4例未报告旅行信息，2例为人传人，5例未有中国旅行史或确诊病例接触史。

尽管美国先行采取了许多遏制措施，包括旅行警告、旅行禁令、入境筛查、接触者追踪等，但仍有多个病例未报告相关的旅行史或接触史，表明社区传播的可能性。论文假设美国疾控中心确定了50%以上的输入性病例。

至于建立新冠病毒的传播范围模型所需的其他一些关键因素，如基本传染数、潜伏期、人际传播时间间隔等，则参照了针对中国病例的现有研究，分别设置为2.1至2.5、6天和7.5天。

在最可能的参数设定下，分析模型显示截至3月1日，若此前采取的措施并未成功减少未确诊病例的传播，美国有9484例感染（90%置信区间，2054到24241）；若措施降低了25%的为确诊病例传播，则感染数字为1043（90%置信区间，107到2474）。

论文表示，在对疾病传播“过度保守的设定”和对美国疾控措施“过度乐观的假设”下，模型依然显示3月1日美国出现千名传染病例。研究团队估计，真实的数字可能介于1000至10000之间。这暗示着，在早期流行阶段就控制COVID-19的机会窗口正在关闭。

论文也引述了一份Bedford实验室在3月2日发表的评估，即新冠病毒已经在西雅图地区社区传播了6周，该地区的感染人数应达到了570例。根据论文的模型，像西雅图这样的社区聚集传播不止一处。

鉴于减少25%的传播，就能将模型评估的感染规模降低至近10%，论文作者建议采取积极的遏制手段，如大规模筛查、减少大规模聚集等。

最后，论文提到，由于新冠病毒在非亚洲群体中的传播动态范围几乎没有参考数据，只能基于中国的流行情况判断，因此也存在高估美国感染人数的可能性。毕竟，传播指数与社会经济、文化、环境因素都有关联。
    '''
    split_sentence(doc)
    # summation output
    result = autosummation(title,doc,model,word_sifs,stop_words,word_idf)
    print(result)
