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
open up `run.sh` and replace the line `/home/ryan/Documents/AIAP/aiap16-Ng Guangren, Ryan-S9303680H/data/lung_cancer.db` with a complete path in your system.
Run `bash run.sh` to start the MLP. your terminal environement should be in a bash to run this programme.

# The logical flow  
<ul>
<li>connect to db and retrieve data in `pd.DataFrame`.   </li>
<li>drop duplicated rows</li>
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
  `.db` format is hard to wrangle data with. SQL is not designed for that, that is why it is extracted into pandas DataFrame format for data wrangling.
  drop duplicated rows because they are the synthetic data from Oversampling as mentioned in the question pape
  columns names cleaning for easier auto complete and indices
  Cleaning of data because there are some unconsistent or nonsensical categories (eg. Male and MALE, "RightBoth" to "Both")
  Some are non smokers or still smoking so those data's start smoking data and stop smoking date should be turned into `0` for easier feature engineering later. types are afterward changed to integer 
  for easier processing
  Some ages are seems to be outliers, 0 and 75percentile range to replace outliers with `0`
  genders have some with "nan", logically it should be replaced with `None` 
  Feature engineering for weight difference and years of smoking as I think they may be useful for modeling.  

  Standard scaling for normalisation
  imputation of missing data
  
# Choice of models
Random Forest Classifier

Gradient Boosting Classifier

Support Vector Decomposition
# Evaluation of models  
Classification report from sklearn for the accuracy, f1 scores and other commonly used performance metrics 
ROC_AUC_score to show classifier performance.  
Confusion Matrix to show Predicted positives and negatives against true positives and negatives 
# Other Considerations
