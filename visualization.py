from collections import Counter
from gensim.models import Word2Vec
import re

def main():
    model = Word2Vec.load('./sample_data/wiki_en_model')

    try:
        num_rows = len(model.vocab)
    except:
        model.vocab = model.wv.vocab
        num_rows = len(model.vocab)

    dim = model.vector_size

    global tensor_out_fn
    global labels_out_fn

    tensor_out_fn = './sample_data/wiki_en_model_%d_%dd_tensors.tsv' % (num_rows, dim)
    labels_out_fn = './sample_data/wiki_en_model_%d_%dd_labels.tsv' % (num_rows, dim)

    try:
        labels_out = open(labels_out_fn, 'w', encoding='utf-8')
    except:
        labels_out = open(labels_out_fn, 'w')

    labels_out.write('word\tlanguage\tcount\n')
    wv_list = []

    counter = {}
    for wd in model.vocab:
        counter[wd] = model.vocab[wd].count
    counter = Counter(counter)
    common = counter.most_common(5000)
    words, _ = zip(*common)

    for wd in words:
        ww = model[wd].tolist()
        assert dim == len(ww)
        assert '\t' not in wd
        wv_list.append(ww)

        try:
            labels_out.write('%s\t%s\t%s\n' % (wd, 'en', model.vocab[wd].count))
        except:
            labels_out.write(('%s\t%s\t%s\n' % (wd, 'en', model.vocab[wd].count)).encode('utf-8'))

    with open(tensor_out_fn, 'w') as fw:
        for i in wv_list:
            fw.write("%s\n" % (str(i).replace(', ', '\t').replace('[', '').replace(']', '')))

    labels_out.close()
    
    
if __name__ =="__main__":
    main()
