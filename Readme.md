# StarSystem

StarSystem is a command line tool based on the [Weka library](http://www.cs.waikato.ac.nz/~ml/weka/) and written in Java. 
Its purpose is to automate best practices in supervised classification experiments. This project was born during my master
thesis in computer science, and the name is a tribute to my supervisor.

## What can StarSystem do for you?

Imagine you were able to collect many features describing a big number of 
events for your phenomenon of interest. This could go from the outcome of a 
basketball game, to diagnosis of patients. You found yourself in the fortunate 
situation of obtaining a very big deal of data, either by extracting them from 
a database, or crawling the www, or using sensors. One could think that "the 
more, the better", but this is not always true in machine learning.

The typical well-conducted supervised classification experiment need to ensure 
two main properties:
1. only the most informative features are used, so to avoid the infamous [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
2. performance on data didn't come out by chance or because of [overfitting](https://en.wikipedia.org/wiki/Overfitting)

Determining the best subset of features is not a trivial task: this could 
require a lot of costly domain knowledge, or just be impossible, because 
knowing how much a feature can influence the outcome of our phenomenon may be 
one of the goals of our experiment. A pre-defined set of features is one more hypothesis introduced 
by the scientist and is prone to induce a bias. The process of selecting an optimal subset of features in 
machine learning is called [feature selection](https://en.wikipedia.org/wiki/Feature_selection) 
and it is best conducted when *agnostic*, that is, without imposing any personal, 
unsupported knowledge on such selection.

As for ensuring stability for found performance, two main techniques exist for 
that: 
[cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), 
and the use of two separate training and test set.

StarSystem puts all this considerations in a single-execution command-line 
tool, implementing the entire experiment in 3 different phases:
1. agnostic feature selection
2. cross-validation evaluation
3. evaluation on test set

The final output of StarSystem is a set of configurations, or couples (Classifier/Set of features) which have been 
proven effective and stable on the data. This of course implies that actual
information is contained into the dataset. Not much will be obtained from poor 
quality data, as from the principle "Garbage in, garbage out".

## Implemented analysis in detail

StarSystem starts with one or more Weka .arff file containing an arbitrary number of 
features together with the outcome of classification. Two of this features are 
particular, as they are used to *sort* and to *slice* the dataset.

### Sorting and slicing the dataset

Sorting feature is usually a date/time feature. An example could
be the date when a sample was collected. When specified, it will cause 
the dataset to be sorted in increasing order. This is done to ensure we don't 
incur in [data snooping](https://en.wikipedia.org/wiki/Data_dredging) during 
test set evaluation, letting algorithms train on parts of the dataset in the 
future and classify the past. For this same reason, sorting feature will be 
removed from the dataset just after it has been used. Be aware that other 
temporal features not used for sorting could be prone to the same problem, 
but won't be removed automatically.

By the term *slicing* the dataset, we mean to divide it into different parts, 
which can be assigned freely to the different phases of the analysis. 
A good example of slicing feature is the season for a dataset of football matches.
If we have six different seasons in our dataset, we could assign two of them 
to feature selection, two to cross-validation and two as test set. Also slicing 
feature will be removed after it has been used.

### Feature selection

Once dataset has been sorted and sliced, feature selection starts. Users must 
decide what is the maximum number of features they wants to retain in the dataset 
and the number of subsets they wants to create. If, for example, we have a total 
of
400 features, we can decide to select the best 100, in subsets of 10, 20, 30... 
100 features.

Informative filters are applied to the entire feature set and the worst features
are removed, so we retain just the maximum number we choose (100 in our example). 
This subset is then passed to wrappers, that is, classifiers that will measure their
accuracy on subsets of the features to find out which one proves itself most informative. 
Each wrapper produces a ranking of the remaining features, and all of these rankings 
are aggregated by mean. Resulting aggregated ranking is used to sort features in the 
dataset, then produce a version of the best *n* features where *n* is a number specified 
in the configuration. In our example we will have subsets of the best 10, 20... 100 features.

Every subset of features so produced is used to create a feature-selected version
of every *slice* of the dataset. This means we will have a 10 features version of 
cross-validation dataset and of test set, a 20 features version and so on.


### Cross validation evaluation

Every combination of (Classifier/Set of features) is evaluated in cross 
validation against a baseline model. By default this is Weka's ZeroR,
that assigns always the most frequent class to each sample, but a model
of choice can be specified in the options. This allows also to benchmark
our classifiers against a less trivial baseline to show which of them
can be told *significantly* better.

At the end of the evaluation, [paired-t test](https://en.wikipedia.org/wiki/Student's_t-test) is used to
retrieve all combinations beating the baseline *significantly*. For each of them, StarSystem
proceeds with evaluation on test set, passing performance value obtained here together with
its *standard deviation*, which will be used as margin of error in the next phase.

### Test set evaluation

Learning schemes which proved themselves better than the baseline are now 
tested against the test set. As training set we will use the portion of 
the dataset assigned to cross-validation.

This phase will determine if performance keeps being 
better than baseline and, most of all, if it *stays the same*: to decide a learning 
scheme is *good and stable*, it must beat the baseline also
on the test set, and its performance measure must be equal to the one obtained in
cross-validation +/- standard deviation. Notice that higher performances surpassing
standard deviation with respect to the baseline are considered equally unstable as 
those being lower.

Evaluation is executed in two different fashions:
1. with retrain: that is, by evaluating each instance, then putting it into the 
dataset and retrain the model
2. without retrain: model is trained only once and used to classify every 
instance in the test set

These two ways of evaluating are meant to spot the eventual influence of sequence on 
classification. Big differences in performances obtained with and without retrain 
should suggest the use of different sequence-aware models for classification, which
StarSystem currently doesn't support.

This is the final phase of the analysis. Once finished, good combinations will be 
available for inspection in final results folder.

## Configuring an experiment

StarSystem allows to configure in great detail the general experimental flow 
described above.
User can decide which filters, wrappers, baseline and classifiers apply to 
the analysis, and many parameters can be set, as the number of iteration 
during feature selection and number of folds for cross-validation.

The experiment can be conducted on many different datasets at a time, which, 
for example, could belong to different national sport championships, or 
different populations of patients.

Evaluation measure can be chosen between accuracy, [precision, recall and f-measure](https://en.wikipedia.org/wiki/Precision_and_recall).

For a better understanding of configuration, please refer to example 
configuration file, containing comments about each option's usage.

Typical steps to start playing with StarSystem are:
1. Clone repo in your favorite directory
2. Import StarSystem in [Eclipse](https://www.eclipse.org/)
3. Include, either by source code or by jar file, the Weka library (3.6, current Debian version)
4. Build a self contained jar file for the entire project
5. Configure the experiment

Be aware that only version 3.6 of Weka is supported. The idea is to keep pace with Debian, so when 
a higher Debian version of Weka will be released, StarSystem will be updated.
Also know that Weka 3.6 suffered a bug between 3.6.11 and 3.6.12 affecting ranking 
of features exploited by StarSystem. For the best safety, please use 3.6 snapshot version of the 
library or versions newer than 3.6.12.

For convenience, this repo will always include a compiled self contained jar version of the latest code.

Once the configuration file is ready, just put your dataset(s) into main dataset folder, by default 
*execution dir*/01-dataset/01-dataset, then start the experiment by typing:

```
java -Xmx<fair-amount-of-memory> -jar StarSystem.jar <conf-file>

```
...and wait! Entire process, especially feature selection's wrappers, can be VERY time and memory
consuming, depending on the number of features and numerosity of the dataset. Also, use of 
computationally expensive models have a big impact on time requirements. Please be patient, or 
set for a less demanding setup for your experiment. This said, StarSystem exploits Java's native 
parallelism and benefits greatly of multi-core execution environments.

At the end of the execution, eventually found configurations can be examined into final results path,
which by default is expected to be in *execution dir*/05-results.