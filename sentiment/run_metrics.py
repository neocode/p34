def run_metrics(sent_dic):
    import pymorphy2, math

    morph = pymorphy2.MorphAnalyzer()

    #Make the dictionary from sent_dic_ru file
    with open(sent_dic) as f:
        sent_learn = f.readlines()
    sent_learn = [x.strip('\n') for x in sent_learn]
    #print(sent_learn)
    n_train = math.trunc(0.8*len(sent_learn))
    n_test = len(sent_learn) - n_train
    #print(n_train, n_test)
    sent_test = sent_learn[n_train:]
    sent_learn = sent_learn[:n_train]
    #print(sent_learn)
    #print(sent_test)

    def norm(word):
        p_obj = morph.parse(word)
        try:
            p = p_obj[0]
            return p.normal_form, p.tag.POS
        except IndexError:
            return None


    def sent_learn_prep(sent):
        #Normalization and stop-words delete from sent_list
        sent_plus = []
        label_plus = []
        sent_minus = []
        label_minus = []

        sent_list = []
        for x in sent:
            #Split into words
            str_list = x.split()

            #List from sentenses (words in list)
            sent_list.append(str_list)

        for sent in sent_list:
            #print(sent)
            one_sent = ''

            if abs(float(sent[-1])) > 0.3:
                for x in sent[:-1]:
                    #print(x)
                    x_norm = norm(x)
                    if x_norm is not None:
                        if x_norm[1] not in ['PREP', 'PRCL', 'CONJ', 'NPRO']:
                            one_sent = one_sent + ' ' + x_norm[0]
                weight = float(sent[-1])
                if weight > 0:
                    weight_norm = 1
                else:
                    weight_norm = -1
                one_sent = one_sent.lstrip()
                #print(one_sent)
                if weight_norm > 0:
                    sent_plus.append(one_sent)
                    label_plus.append(weight_norm)
                else:
                    sent_minus.append(one_sent)
                    label_minus.append(weight_norm)
        #print(sent_list_norm)
        #print(label_list)
        return (sent_plus, label_plus), (sent_minus, label_minus)

    sent_p, sent_m = sent_learn_prep(sent_learn)
    #print(sent_p, sent_m )

    from nltk.corpus import stopwords
    stopWords = stopwords.words('russian')

    from sklearn.feature_extraction.text import CountVectorizer
    vectoriz_p = CountVectorizer(stop_words = stopWords)
    vectoriz_m = CountVectorizer(stop_words = stopWords)

    #Bags of words making
    sent_vect_p = vectoriz_p.fit_transform(sent_p[0])
    sent_vect_m = vectoriz_m.fit_transform(sent_m[0])

    import numpy as np
    words_p = vectoriz_p.get_feature_names()
    counts_p = np.asarray(sent_vect_p.sum(axis=0)).ravel()
    words_counts_p = dict(zip(words_p, counts_p))

    words_m = vectoriz_m.get_feature_names()
    counts_m = np.asarray(sent_vect_m.sum(axis=0)).ravel()
    words_counts_m = dict(zip(words_m, counts_m))

    #Intersection between two sets (with +1 and -1 sentiment)
    words_intersect = set(words_p).intersection(set(words_m))
    #print(words_intersect)

    #Expanding the list of stop words
    for x in words_intersect:
        if abs(words_counts_p[x] - words_counts_m[x]) <= 1:
            stopWords.append(x)
    #print(stopWords)

    #Making the joined bag of words
    vectoriz = CountVectorizer(stop_words = stopWords, min_df = 0.001, max_df = 0.999)
    sent_vect = vectoriz.fit_transform(sent_p[0] + sent_m[0])
    #List of words - coordinates of the vector
    words = vectoriz.get_feature_names()
    counts = np.asarray(sent_vect.sum(axis=0)).ravel()
    words_counts = dict(zip(words, counts))
    #print(words)

    #Making vectorizer for learning SVM (on the base of words list)
    cv = CountVectorizer(vocabulary=words)
    vect_sent = cv.fit_transform(sent_p[0] + sent_m[0])
    vect_label = np.array(sent_p[1] + sent_m[1])

    from sklearn.svm import SVC # "Support Vector Classifier"
    clf = SVC(kernel='linear')
    clf.fit(vect_sent, vect_label)

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
            for x in sent[:-1]:
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

    sent_list_norm = sent_prep(sent_test)
    #print(sent_list_norm)
    #cv = CountVectorizer(vocabulary = dic)
    vect_sent = cv.fit_transform(sent_list_norm)
    #print(vect_sent)

    vect_label_test = []
    for x in vect_sent:
        vect_label_test.extend(clf.predict(x).tolist())

    vect_label_true = []
    for x in sent_test:
        str_list = x.split()
        if float(str_list[-1]) >= 0:
            vect_label_true.append(1)
        else:
            vect_label_true.append(-1)

    #for i in range(len(sent_test)):
    #    print(sent_test[i], vect_label_test[i])

    #print(vect_label_true)

    yes = 0
    no = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(n_test):
        if vect_label_test[i] ==  vect_label_true[i]:
            yes += 1
            if vect_label_test[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            no += 1
            if (vect_label_test[i] == 1) and (vect_label_true[i] == -1):
                FP += 1
            else:
                FN += 1
    PR = yes/n_test
    RC = yes/yes
    F1 = (2*PR*RC)/(PR+RC)
    ACC = (TP + TN)/(TP + FP + TN + FN)
    print('PR =', PR*100, '%')
    print('RC =', RC*100, '%')
    print('F1 =', F1)
    print('TP =', TP, 'TN =', TN, 'FP =', FP, 'FN =', FN)
    print('ACC =', ACC)

#run_metrics('sent_dic_ru__.txt')
