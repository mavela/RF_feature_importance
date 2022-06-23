import sys
import gzip
import numpy as np
import re
from sklearn.metrics import classification_report
import sklearn.feature_extraction
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from joblib import dump, load

#from sklearn.inspection import permutation_importance

topic_vectors = []
score_vectors = []

regs = ["Narrative", "Opinion", "Informational description (or explanation)", "Informational persuasion", "Interactive discussion", "Lyrical", "Spoken", "How-to/instructions"]

#How-to/instructions 


c = 0
for index, t in enumerate(sys.stdin):
#    print("fuc")
    orig = t
    c+=1
 #   print("c", c)
    t=t.strip().split(" ")
#    print()
#    print("orig", t)
    topic_vector = []
    score_vector = []
    for number in t[-180:]:
        number=number.replace("[", "")
        number=number.replace("]", "")
        number=number.replace(",", "")
        number=number.strip()
        topic_vector.append(float(number))
 #   print("t",topic_vector)
    score_list = " ".join(t[:-180])
    score_list=score_list.strip().split(",")

    if not score_list[0].startswith("id"):
        print("ERROR WITH SCORELIST")
    else:
        count = 0
        for item in score_list:
            if item.startswith("id"):
                continue
            else:
 #                   print("orit", item)
                    item=item.replace("predicted registers:", "")
                  #  print("2", item)
                    item=item.strip().split("=")
  #                  print("item",item)
                    if len(item)==1:
                        continue
                    else:
                        if item[0].strip() in regs:
   #                         print("JEE")
                            score_vector.append(float(item[-1]))
                        
#        print("score_vector", score_vector)
#        score_vector = score_vector[0]
        if len(score_vector) != 1:
            print("ERROR, NOT 1 SCORE", index)
            print(score_vector)
            print("orig", orig)
            print("score_list", score_list)
        else:
            score_vector = score_vector[0]
#            print("score_Vector", score_vector)
            score_vectors.append(score_vector)
 #   print("final", score_vectors)
    if len(topic_vector) != 180:
        print("ERROR WITH TOPIC VECTOR")
    else:
        topic_vectors.append(topic_vector)

if len(topic_vectors) != len(score_vectors):
    print("ERROR! NOT SAME LEN TOPIC AND SCORE VECS")

print("### data in", flush=True)
#print(topic_vectors[2])
#print(score_vectors[2])
    

def data_iterator(f):
    for token in f:
        yield token

def tokenizer(txt):
    """Simple whitespace tokenizer"""
    return txt.split()


topic_array = np.array(topic_vectors)

raja_arvot = []
for q in [0.25,0.5,0.75]:
    score_array = np.quantile(score_vectors,q)#np.arange(0.01,1,0.01))
    raja_arvot.append(score_array)

bin_vecs = []
#print("score_vector", score_vectors)
for val in score_vectors:    
    if val <= float(raja_arvot[0]):
        bin_vecs.append("low")
    if float(val) > float(raja_arvot[0]):
        if float(val) <= float(raja_arvot[1]):
            bin_vecs.append("mid_low")
    if val > float(raja_arvot[1]):
        if val <= float(raja_arvot[2]):
            bin_vecs.append("mid_high")
    if float(val) > float(raja_arvot[2]):
        bin_vecs.append("high")  

if len(score_vectors)!= len(bin_vecs):
    print("ERROR IN BIN VECS")
#    print("score_array", "q=", q, score_array, flush=True)
print()
feats_train, feats_test, scores_train, scores_test = train_test_split(topic_array, score_vectors, test_size=0.2, stratify=bin_vecs)

#feats_train

print("### feats done", flush=True)
feat_names = np.array(list(range(1,180)))
for n_estimator in [1000]:
    for mx in  np.arange(13,180,20):
        for mn in [5,6,7,8,9,10]:
            
            classifier=RandomForestRegressor(n_estimators = n_estimator, max_features=mx, max_depth=mn, random_state=0)
            classifier.fit(feats_train,scores_train)
            print("n_esti="+ str(n_estimator) + "\t" + "max=" + str(mx) + "\t" + "mn=" + str(mn) + "\t" + "score=" +  str(classifier.score(feats_test,scores_test)), flush=True)
#            dump(classifier, "randomregre_model.joblib") # uncomment if you want to save the model


