from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import gzip
import numpy as np
import re
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

labels_train=[]
labels_test=[]

label_feats=[]
test = []
train = []


for t in sys.stdin:    
    label_feats.append(t)

dict = {}

for s in (label_feats):
    clas,sent=s.split(u"\t",1)
    if clas not in dict:
       dict[clas]=0
    dict[clas]+=1
    # tarkista, onkosanakirjassa, jos ei lisää 

dict2={}

for count, t in enumerate(label_feats):
    x,feat=t.split(u"\t",1)
    if x not in dict2:
        dict2[x]=0
    dict2[x]+=1
    if dict[x]*0.2 > dict2[x]:
        labels_test.append(x)
        feat = [float(x) for x in feat.split(" ")]
        test.append(feat)
    else: 
        x,feat=t.split(u"\t",1)
        labels_train.append(x)
        feat = [float(x) for x in feat.split(" ")]
        train.append(feat)

print("### feats done", flush=True)
feat_names = np.array(list(range(1,180)))
for n_estimator in [int(sys.argv[1])]:
    for mx in  [int(sys.argv[2])]:#13,180,20):
        for mn in [5,6,7,8,9,10]:

            classifier=RandomForestClassifier(n_estimators = n_estimator, max_features=mx, max_depth=mn, random_state=0)
            classifier.fit(train,labels_train)
            print("n_esti="+ str(n_estimator) + "\t" + "max=" + str(mx) + "\t" + "mn=" + str(mn) + "\t" + "score=" +  str(classifier.score(test,labels_test)), flush=True)


"""
for c in [500]:#,0.4,0.5,0.6,0.8,1,5,10,20]:#0.001, 0.00001, 0.0001, 0.0005, 0.001,0.005,0.1,0.5,1]:#,5,10]:
    print("c", c)
    classifier=SVC(C=c,kernel='linear',shrinking=False)#,class_weight="balanced")
#    classifier=LinearSVC(C=c,class_weight="balanced")
    classifier.fit(d,labels_train)

#    print( "Accuracy is", classifier.score(d_test,labels_test))
#    print("C is", c)
    #print "REAL ONES", labels_test
    labels_test_true = labels_coca
    labels_test_pred = classifier.predict(d_test)
    print(classification_report(labels_test_true, labels_test_pred))
"""
