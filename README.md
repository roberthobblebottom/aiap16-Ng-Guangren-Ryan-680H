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
open up `run.sh` and replace the line `/home/ryan/Documents/AIAP/aiap16-Ng Guangren, Ryan-S9303680H/data/lung_cancer.db` with a complete path as reflected in your system.
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

# Overview key findings from `eda.ipynb`
  
# Feature Processing
|Feature    |Reason     |
---|---
|drop duplicated rows| because they are the synthetic data from Oversampling as mentioned in the question paper. |
|columns names cleaning| for easier auto complete and indices|
|Cleaning of data| there are some unconsistent or nonsensical categories (eg. Male and MALE, "RightBoth" to "Both")|
|non year data to "0" and these features types is changed to int| Some are non smokers or still smoking so those data's start smoking data and stop smoking date should be turned into `0` instead of `Not Applicable` or `Still Smoking` for easier feature engineering later. for easier processing
 | Replace outliers with 0 in age| Some ages are seems to be outliers, 0 value point and 75 percentile range to replace outliers with `0`
 |Replacing gender `"nan"` string with `None`| genders have some with "nan", logically it should be replaced with `None` 
|Feature engineering for weight difference and has_history_of_smoking| as  they may be useful for modeling.  
 | Standard scaling |for normalisation
 | imputation |of missing data as shown in the `eda.ipynb`
 |SelectKBest| for features and label based, supervised feature selections for the reduction of dimensions to mitigate dimension reduction
  
# Choice of models
Ensemble models can be useful as they are more robust and generalised and often able to gain better results over single estimators like decision trees. These two methods below are very popular.

Random Forest Classifier can help to determine feature importance esily. easier to tun than gradient boosting, harder to overfit than gradient boosting.

Gradient Boosting Classifier - helps improve on the error on the previous tree trained got wrong. More flexible than logistic regression.Histogram variant is used for dataset with more than 10,000 samples as binning can help to speed up the gradient tree boosting through tghe reudction of number of values from continuous range to discrete range  and can evenx improving preformance of the classifier.

Support Vector Mechanism
# Evaluation of models  
Classification report from sklearn for the accuracy, f1 scores and other commonly used performance metrics 
ROC_AUC_score to show classifier performance.  
Confusion Matrix to show Predicted positives and negatives against true positives and negatives 
# Other Considerations
the script has been logically seggregated in their own respective module/classes. This is for enhanced readability, maintainability and extensibility.  

different dataset may be used in the future, `preprocess.py` should be changed accordingly to the needs of the new datasets' veracity. For example maybe there are more features that has inconsitent categorical values eg, "yes" and "YES" so those have to be done handled in the future. `pipeline.py` may have it's pipeline changed or even user configurable during execution in case a different preprocessing steps are taken.
