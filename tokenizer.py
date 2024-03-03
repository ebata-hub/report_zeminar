import MeCab

tagger = MeCab.Tagger()
stop_words = ['、', '。','する','よう','できる','もの','こと']

#stop_words_ja = ['もの', 'こと', 'とき', 'そう', 'たち', 'これ', 'よう', 'これら', 'それ', 'すべて','の','ん']

def tokenize(text):
    node = tagger.parseToNode(text)
    
    tokens = []
    while node:
        features = node.feature.split(',')
        if features[0] != 'BOS/EOS':
            if features[0] not in ['助詞', '助動詞']:
#            if features[0] not in ['']:
                token = features[6] if features[6] != '*' else node.surface
                if token not in stop_words:
                    tokens.append(token)
                
        node = node.next

    return tokens
