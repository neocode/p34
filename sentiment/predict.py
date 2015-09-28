def predict(model, vocabl):

    from sklearn.externals import joblib
    import re
    clf = joblib.load(model)

    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()

    def norm(word):
        p_obj = morph.parse(word)
        try:
            p = p_obj[0]
            return p.normal_form, p.tag.POS
        except IndexError:
            return None


    def sent_prep(sent):
        #Normalization and stop-words delete from sent_list
        sent_list_norm = []
        sent_list = []
        for x in sent:

            #Split into words
            str_list = x.split()

            #List from sentenses (words in list)
            sent_list.append(str_list)
        for sent in sent_list:
            #print(sent)
            one_sent = ''
            for x in sent:
                    #print(x)
                    x_norm = norm(x)
                    #print(x_norm)
                    if x_norm:
                        if x_norm[1] not in ['PREP', 'PRCL', 'CONJ', 'NPRO']:
                            one_sent = one_sent + ' ' + x_norm[0]
            one_sent = one_sent.lstrip()
            #print(one_sent)
            sent_list_norm.append(one_sent)
        return sent_list_norm

    def raw_cut(text):
        #Delete digits
        text = re.sub(r"\b\d+\b", "", text)
        #Split text into sentences and remove empty elements from list
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = list(filter(None, sentences))
        return sentences

    with open(vocabl) as f:
        words = f.readlines()
    words = [x.strip('\n') for x in words]
    #print(words)
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(vocabulary=words)

    text = input("Please enter some text: ")
    sentences = raw_cut(text)
    sent_list_norm = sent_prep(sentences)
    vect_sent = cv.fit_transform(sent_list_norm)
    vect_label_test = []

    for x in vect_sent:
        vect_label_test.extend(clf.predict(x).tolist())

    for i in range(len(sentences)):
        print(sentences[i], vect_label_test[i])

    return 'Prediction complete'

#predict('./model/model.pkl', './model/Words.txt')