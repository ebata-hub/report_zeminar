from tokenizer import tokenize
import neologdn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from wordcloud import WordCloud
import pandas as pd
import nlplot
import plotly
from plotly.subplots import make_subplots
from plotly.offline import iplot
import matplotlib.pyplot as plt
import pyvis
from pyvis.network import Network
import itertools
import collections

def normalize_text(text):
    text_n = neologdn.normalize(text)
    return text_n

def calc_bow(tokenized_texts):  # <2>
    # Build vocabulary <3>
    vocabulary = {}
    for tokenized_text in tokenized_texts:
        for token in tokenized_text:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    n_vocab = len(vocabulary)

    # Build BoW Feature Vector <4>
    bow = [[0] * n_vocab for i in range(len(tokenized_texts))]
    for i, tokenized_text in enumerate(tokenized_texts):
        for token in tokenized_text:
            index = vocabulary[token]
            bow[i][index] += 1

    return vocabulary, bow

def co_occurrence_network(sentences):
    texts = [s for s in sentences.split('。') if s]
    df = pd.DataFrame(texts, columns=['text'])
    df['words'] = df['text'].apply(tokenize)
    print(df)

    # nlplot
    npt = nlplot.NLPlot(df, target_col='words')
    stopwords = npt.get_stopword(top_n=0, min_freq=0)

    # unigram
    fig_unigram = npt.bar_ngram(
        title='単語の登場回数',
        xaxis_label='単語の登場回数',
        yaxis_label='単語',
        ngram=1,
        top_n=50,
        stopwords=stopwords,
        save = True,
    )
    fig_unigram.show()

    # 共起分析
    npt.build_graph(stopwords=stopwords, min_edge_frequency=1)
    fig_co_network = npt.co_network(title='共起ネットワーク', save=True)
    iplot(fig_co_network)
    
    return

def get_key_with_value_in_dict(dict, value):
    key = ''
    for k, v in dict.items():
        if v == value:
            key = k
    return key

def kyoki_word_network():
    from pyvis.network import Network
    import pandas as pd

    #got_net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    got_net = Network(height="1000px", width="95%", bgcolor="#FFFFFF", font_color="black", notebook=True)

    # set the physics layout of the network
    #got_net.barnes_hut()
    got_net.force_atlas_2based()
    got_data = pd.read_csv("kyoki.csv")[:150]

    sources = got_data['first']#count
    targets = got_data['second']#first
    weights = got_data['count']#second

    edge_data = zip(sources, targets, weights)

    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]

        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        got_net.add_edge(src, dst, value=w)

    neighbor_map = got_net.get_adj_list()

    # add neighbor data to node hover data
    for node in got_net.nodes:
        node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])

    got_net.show_buttons(filter_=['physics'])
    return got_net


#--------------------
# Main
#--------------------
with open('report1.txt', 'r',encoding='UTF-8') as f1:
    text1 = f1.read()
    
with open('report2.txt', 'r',encoding='UTF-8') as f2:
    text2 = f2.read()

print(text1)
text1n = normalize_text(text1)
print(text1n)
tokens1 = tokenize(text1n)
print(tokens1)

print(text2)
text2n = normalize_text(text2)
print(text2n)
tokens2 = tokenize(text2n)
print(tokens2)

texts = [text1, text2]
vectorizer = CountVectorizer(tokenizer=tokenize)
vectorizer.fit(texts)
bow = vectorizer.transform(texts).toarray()
print(bow)
print(bow.shape)

vocabulary = vectorizer.vocabulary_
print(vocabulary)
print(len(vocabulary))

bow_sort = np.sort(bow, axis=1)[:, ::-1]
print(bow_sort)

bow_argsort = np.argsort(bow)[:, ::-1]
print(bow_argsort)

row, col = bow_argsort.shape
print(row, col)
vocabulary_sort = [[0]*col for i in range(row)]
index_sort = [[0]*col for i in range(row)]
for i in range(row):
    for j in range(col):
        index = bow_argsort[i][j]
        #print(i, j, index)
        #print(index_sort[0][j], i)
        index_sort[i][j] = index
        #print(index_sort[0][j])
        
        key = get_key_with_value_in_dict(vocabulary, index)
        #print(i, j, index, index_sort[i][j], index_sort[0][j], key)
        vocabulary_sort[i][j] = key
        
print(vocabulary_sort)
print(index_sort)

print('***** 単語の頻度  *****')
for i in range(row):
    for j in range(col):
        print(i, ',', bow_sort[i][j], ':', vocabulary_sort[i][j])
        

# Word Cloud
wc = WordCloud(width=480, height=320, background_color="white",
               font_path="/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc")
wc.generate(" ".join(tokens1))
wc.to_file('wordcloud1.png')
wc.generate(" ".join(tokens2))
wc.to_file('wordcloud2.png')


#
# nlplot
#
co_occurrence_network(text1)
co_occurrence_network(text2)


#texts = [s for s in text1.split('。') if s]
#texts = [s for s in text2.split('。') if s]

#df = pd.DataFrame(texts, columns=['text'])
#df['words'] = df['text'].apply(tokenize)

#print(df)

# nlplot
#npt = nlplot.NLPlot(df, target_col='words')
#stopwords = npt.get_stopword(top_n=0, min_freq=0)

#fig_unigram = npt.bar_ngram(
#    title='uni-gram',
#    xaxis_label='word_count',
#    yaxis_label='word',
#    ngram=1,
#    top_n=50,
#    stopwords=stopwords,
#    save = True,
#)
#fig_unigram.show()

# 共起分析
#npt.build_graph(stopwords=stopwords, min_edge_frequency=1)
#fig_co_network = npt.co_network(title='Co-occurrence network', save=True)
#iplot(fig_co_network)











#
# pyvis
#
#sentences = [w for w in text1.split("。")]
#sentence_combinations = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
#sentence_combinations = [[tuple(sorted(words)) for words in sentence] for sentence in sentence_combinations]
#target_combinations = []
#for sentence in sentence_combinations:
#    target_combinations.extend(sentence)

#ct = collections.Counter(target_combinations)
#ct.most_common()[:10]

#pd.DataFrame([{'first' : i[0][0], 'second' : i[0][1], 'count' : i[1]} for i in ct.most_common()]).to_csv('kyoki.csv', index=False)
#got_net = kyoki_word_network()
#got_net.show("kyoki.html")
