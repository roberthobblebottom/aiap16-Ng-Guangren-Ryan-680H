Ng Guangren, Ryan S9303680H

#Repo Directory

```
.github
src/
  main.py
  pipeline.py
  preprocess.py
  sqlite.py
data/
  lung_cancer.db
README.md
eda.ipynb
requirements.txt
run.sh

```
The logical flow:  
connect to db and retrieve data in `pd.DataFrame`.  
Do columns and indices names cleaning  
Enter into pipeline for Random Forest Classifier, Gradient Boosting Classifier, Support Vector Mechanism from `scikit-learn` library ( or in short sklearn).  
The pipeline function:  
  numerical features names and categorial feature names will be first extracted.   
  the sklearn `Pipeline` will be build with:  
    Cleaning of data using my custom sklearn estimator class from `preprocess.py`  
    Feature engineering using my custom sklearn estimator class from `preprocess.py`  
    `ColumnTransformer` for only encoding categorial features into numerics using `OrdinalEncoder`.
    `StandardScaling` to normalise my now fully numerical features.
    `SimpleImputer` to impute missing data using the default mean strategy.   
    `SelectKBest` to reduce the number of features.
    `model` for the actually classification to happen
  `train` and `evaluation` will be split with `train_test_split`
  `features` and `target` will be further split from `train` and `evaluation`
  pipeline is be then fitted
  pipeline is then predicted using `evaluation_x`
  Returns a dictionary containing model, `classifcation_report`, `roc_auc_score` and `confusion_matrix`
    
    
