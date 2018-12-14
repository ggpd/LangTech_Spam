# Language Technology Spam Classifier
This project is a binary classifier of spam based on the [SpamAssassin](https://spamassassin.apache.org/old/publiccorpus/) dataset. It uses tools such as nltk and scikit learn to construct this classifier.

## Usage
This projects dependencies are managed through [Pipenv](https://pipenv.readthedocs.io/en/latest/). You must first install this program before usage.

Install dependencies using:

`pipenv install`

Next you must download and preprocess the training data, to do this you must run the `data` sub-command and feed it a directory to download the data to and a place to output data vectors in csv format. In the following command we output the data into the directory `data` and output the data vectors to the current directory, '`.`'. This command must be run in the `spam_classifier` subdirectory.

`pipenv run python cli.py data data .`

After the data has been downloaded you can train the classifier with the choosen classifier. 
Where nb is Naive Bayes, svm is Support Vector Machines, knn is K Nearest Neighbors and rf is Random Forest.

`pipenv run python cli.py train train.csv class.pkl [nb|svm|knn|rf]`

Lastly the classifier can be tested with the following command where it will output
the statistics and output an error.csv of locations of misclassifications.

`pipenv run python cli.py test test.csv class.pkl error.csv`
