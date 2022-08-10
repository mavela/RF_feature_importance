# RF_feature_importance

## Regressor

(module load python-data)

cat toydata_regressor.tsv | python3 randomforest_regressor.py 500 13 # n_estimators, max, to be tested w diff values

Performance and parameters for the final model: 
n_esti=1500     max=173 mn=13   score=0.11405312394736267

Feats and final model available at http://dl.turkunlp.org/register/topicmodeling/

## Classifier

(module load python-data)

cat toydata_classes.tsv | python3 randomforestclassifier.py 500 13 # n_estimators, max, to be tested w diff values

## Save model

add  dump(classifier, "[modelname].joblib") 
