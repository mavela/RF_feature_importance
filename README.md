# RF_feature_importance

## Regressor

(module load python-data)

cat toydata_regressor.tsv | python3 randomforest_regressor.py 500 13 # n_estimators, max, to be tested w diff values

## Classifier

(module load python-data)

cat toydata_classes.tsv | python3 randomforestclassifier.py 500 13 # n_estimators, max, to be tested w diff values

## Save model

add  dump(classifier, "[modelname].joblib") 
