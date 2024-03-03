Ng Guangren, Ryan S9303680H

# Repo Directory

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
# Instruction
Run `bash run.sh` to start the MLP. your terminal environement should be in a bash to run this programme.

# The logical flow  
<ul>
<li>connect to db and retrieve data in `pd.DataFrame`.   </li>
<li>columns and indices names cleaning   </li>
<li>Enter into pipeline for Random Forest Classifier, Gradient Boosting Classifier, Support Vector Mechanism from `scikit-learn` library ( or in short sklearn).   </li>
<li>The pipeline function:  
  <ul>
  <li>numerical features names and categorial feature names will be first extracted.    </li>
  <li>the sklearn `Pipeline` will be build with:  
    <ul>
    <li>Cleaning of data using my custom sklearn estimator class from `preprocess.py`   </li>
    <li>Feature engineering using my custom sklearn estimator class from `preprocess.py`   </li>
    <li>`ColumnTransformer` for only encoding categorial features into numerics using `OrdinalEncoder`. </li>
    <li>`StandardScaling` to normalise my now fully numerical features. </li>
    <li>`SimpleImputer` to impute missing data using the default mean strategy.    </li>
    <li>`SelectKBest` to reduce the number of features. </li>
    <li>`model` for the actually classification to happen </li>
    </ul>
     </li>
  <li>`train` and `evaluation` will be split with `train_test_split` </li>
  <li>`features` and `target` will be further split from `train` and `evaluation` </li>
  <li>pipeline is be then fitted </li>
  <li>pipeline is then predicted using `evaluation_x` </li>
  <li>Returns a dictionary containing model, `classifcation_report`, `roc_auc_score` and `confusion_matrix` </li>
  </ul>
  </li>
</ul>

# Overview key findings
  
# Feature Processing
  
# Choice of models
    
# Evaluation of models  

# Other Considerations
