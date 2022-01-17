# Privacy and Fairness in Recommender Systems 

## Instructions

1. clone the repository
```shell script
git clone https://github.com/marinimau/privacy_and_fairness_in_recsys && cd privacy_and_fairness_in_recsys
```

2. create a virtual environment and activate it
```
python3 -m venv venv
source venv/bin/activate
```

3. install requirements 
```
pip3 install -r requirements.txt
```

4. download the data
```
python3 setup.py
```

5. run the experiments
```
python3 main.py
```

## Attributes

### urls

```
urls = {
    "movielens1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
}
```

a dict that contains the urls required in the source, in this case we have only the url to download movielens1M

### classifier attributes

```
classifiers = ['random-forest', 'logistic-regression']
```
the list of classifier to use

```
classifier_models = {
    "random_forest": RandomForestClassifier(),
    "logistic_regression": LogisticRegression(),
    "naive_bayes": GaussianNB()
}
```
a dict that contains the Sklearn model for each classifier

```
classifier_params = {
    'random_forest': {
        'bootstrap': [False, True],
        'max_features': ['auto'],
        'n_estimators': [50, 700, 1800]
    },
    'logistic_regression': {
        'solver': ['lbfgs'],
        'penalty': ['l2', None],
        'C': [1.0],
        'random_state': [np.random.RandomState(0)],
    },
    'naive_bayes': {
        'priors': [[0.5, 0.5]]
    }
}
```
a dict that contains, for each classifier, the params for the grid search

```
trade_off_param_range = {
    'random_forest': np.arange(200, 700, 100),
    'logistic_regression': [0.001, 0.05, 0.1, 0.5, 1.0, 10.0],
    'naive_bayes': [0.001, 0.05, 0.1, 0.5, 1.0, 10.0]
}
```
a dict that contains, for each classifier the x ticks for the learning and validation curves plot

```
trade_off_param_name = {
    'random_forest': 'n_estimators',
    'logistic_regression': 'C',
    'naive_bayes': 'priors'
}
```
a dict that contains, for each classifier, the x axis title for the learning and validation curves plot

```
classifier_evaluation_plot = False
```
a flag that indicates if we need to generate learning and validation curves

```
current_trade_off_file_name = ''
```
the path to save the learning and validation curves (it is edited a run time)

```
SHOW_PLOT = False
```
a flag tha indicates if we need to see plot a run time (plots are still saved)

## Recommender

```
best = [10, 20, 50]
```
the list of cutoffs for the recs

```
recs_file_names = [
    'user-knn',
    'item-knn',
    'most-pop',
    'bprmf',
    'neumf',
    'wrmf',
    'lightgcn',
    'multidae',
]
```
a list with names of relevance files (recs are extracted from the relevance files)

```
ignore_embeddings = [
    'most-pop',
    'neumf',
    'lightgcn',
    'multidae',
]
```
if a relevance file name is in this list we ignore embedding experiments for that classifier. Experiment are ignored also
if the file is not found.

```
dataset_names = [
    'observation',
    'observation-no-score',
    'most-pop_relevance',
    'most-pop_embeddings',
    'most-pop_classification_best_10',
    'most-pop_classification_best_20',
    'most-pop_classification_best_50',
    'item-knn_relevance',
    'item-knn_embeddings',
    'item-knn_classification_best_10',
    'item-knn_classification_best_20',
    'item-knn_classification_best_50',
    'user-knn_relevance',
    'user-knn_embeddings',
    'user-knn_classification_best_10',
    'user-knn_classification_best_20',
    'user-knn_classification_best_50',
    'bprmf_relevance',
    'bprmf_embeddings',
    'bprmf_classification_best_10',
    'bprmf_classification_best_20',
    'bprmf_classification_best_50',
    'neumf_relevance',
    'neumf_embeddings',
    'neumf_classification_best_10',
    'neumf_classification_best_20',
    'neumf_classification_best_50',
    'wrmf_relevance',
    'wrmf_embeddings',
    'wrmf_classification_best_10',
    'wrmf_classification_best_20',
    'wrmf_classification_best_50',
    'lightgcn_relevance',
    'lightgcn_embeddings',
    'lightgcn_classification_best_10',
    'lightgcn_classification_best_20',
    'lightgcn_classification_best_50',
    'multidae_relevance',
    'multidae_embeddings',
    'multidae_classification_best_10',
    'multidae_classification_best_20',
    'multidae_classification_best_50',
]
```
list that contains the name of the experiments, names ara generated automatically in the source, but we can perform a validation
of the experiment name using this file. (like a test)

```
required_experiments = {
    'observation': False,
    'embeddings': True,
    'recs': False
}
```
a dict that indicates the required experiments

```
required_obfuscated = {
    'observation': True,
    'embeddings': True,
    'recs': True
}
```
a dict that indicates where we need the obfuscated data

```
obfuscation_path = [
    'pred/',
    'avg/',
    'filtered/'
]
```
a list that contains the obfuscated file paths (one for each method)

```
obfuscation_method_index = 2
```
the index of "obfuscation_path" to select the obfuscation method

```
obfuscated_method = obfuscation_path[obfuscation_method_index] if required_obfuscated['observation'] else ''
```
generates the path of the input files: '' if we not need obfuscation; or the current obfuscation_path if we need obfuscated data

### Sensitive attributes inference

```
label_names = ['age', 'gender']
```
the name of the sensitive attributes

```
age_labels = ['1', '56', '25', '45', '50', '35', '18']
```
a list of age label_names for movielens 1M, (not required if we binarize age)

### Classifier evaluation

```
metrics = ['balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'accuracy_score']
```
a list that contains the metrics we are using

### Dataset

```
data_root_list = ['movielens1M', 'lastfm1k']
```
a list that contains the dataset. The organization of the input (interactions, embeddings, recs), results etc. is the same for each dataset.

```
data_root = data_root_list[0]
```
the current dataset

### Time split

```
time_split_cutoffs = [.25, .50, .75, 1]
```
a list of cutoffs for the time splitting

```
time_split_current_cutoff = time_split_cutoffs[3]
```
the selected cutoff for the experiments

```
time_split_cutoffs_fixed = [5, 10, 20]
```
a list of fixed cutoffs for the time splitting

```
time_split_current_cutoff_fixed = time_split_cutoffs_fixed[2]
```
the selected cutoffs for the experiments if we require fixed cutoffs

```
perform_time_splitting = False
```
a flag that indicates if we need to perform time splitting experiments (instead of using the entire dataset)

```
fixed_time_splitting = False
```
a flag that indicates if we need to use fixed time split data (if False we use percentage time splitting)

### Classification preprocessing

```
balance_data = True
```
a flag that indicates if we need to balance the dataset

```
normalize_data = False
```
a flag that indicates if we need to normalize data (sklearn.preprocessing.normalize)

```
lite_dataset = False
```
if true we use the first only "lite_dataset_size" users for the experiments

```
lite_dataset_size = 100
```
indicates how many users use for the experiments

```
test_set_size = 0.3
```
the size of the test set, the size of the training set is 1 - test_set_size

```
maintain_order_in_train_test_split = False
```
a flag that indicates if we need to maintain order in the training and test data

### Filtering method

```
filtering_sampling_percentages = [1, 0.8, 0.6, 0.4]
```
the weight for each partition (the first is the one with the most recent interactions)

```
n_subset_for_filtering = len(filtering_sampling_percentages)
```
the number of partitions (it is the length of "filtering_sampling_percentages")

### General

```
VERBOSE = True
```
verbose flag

```
DEBUG = False
```
not perform classification, allows you to quickly test the paths and the execution process of all experiments


## Author

[Mauro Marini](https://github.com/marinimau)