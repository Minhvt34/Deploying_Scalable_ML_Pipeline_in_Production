# Model Card

This model card describes a model that was trained as a part of Udacity's fourth project, Deploying a Scalable ML Pipeline in Production, for the DevOps nanodegree.


## Model Details

we created the model. It is a feedforward neural network model trained using tensorflow framework.
It has 4 layers with 512, 256, 128, and 128 neurons using relu activationcand two dropout layers for regularization. This model has been put in to dvc pipeline as well.
See dvc.yaml and params.yaml for final hyperparameters and different experiment results.
Before the first layers, a batch normalization layer is added.
A softmax layer with 2 neurons was added to the end to match the shape of labels, a binary classifier.

It was trained using Adam's optimizer with a learning rate of 4e-4 for 10 epochs with batch size of 16.

The model has been unit tested for its input and output data shapes.

## Intended Use

This model is trained to predict if a person's salary is below or above $50k.

## Training Data

    The training data for this model was obtained from https://archive.ics.uci.edu/ml/datasets/census+income
    The target class for this model was "salary".

    The original data set has 32561 rows and 15 features identifying each user in the dataset. Dataset was splitted 80-20 for train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

    Model was evaluated on randomly selected, 20% of the dataset, that the model have no seen. Model was then evaluated on slices of the test dataset mentioned below.  
                                "workclass",
                                "education",
                                "marital-status",
                                "occupation",
                                "relationship",
                                "race",
                                "sex",
                                "native-country"
## Metrics

    Three metrics were used to evaluate this model, Precision, Recall, and F-beta score.

    The trained model performance over all slices of the test dataset is as following:

     Precision is 0.69 Recall is 0.68 and F-Beta Score is 0.68

## Ethical Considerations

    At a glance, profiling the training dataset reveals several critical issues that must be taken into account before deploying such model in production.

### Dataset
    Number of features are highly correlated out of the box.
    The main correlation is between eduction, eduction-num, and marital status.
    This dataset is racially biased, with White being the majority 85% of the dataset.
    The majority of the user are male, over 75% and most users, around 80%, make less than $50k.
    Almost 90% of the dataset are Private work classes and over 90% are from United States.

### Model
    Model has low f-beta scores for specific categories under race, for example, Amer-Indian-Eskimo or Canada. A similar obeservation can be seen udner education for Bachelors and Prof-school. This tells us that the model can make unethical decisions based on someone's race or education.

## Caveats and Recommendations

    It is important to note that this model should not be deployed in production or even considered for making decisions about giving someone a loans. The dataset this model was trained is biased, as discussed under Ethical Considerations/Dataset. This model is more likely to make a decision for a person of White race. Given how correlated the labels in the datasets are, removing race from the dataset would not solve this problem as this feature may have very well been bleed into other features. For more detailed analysis of the model see "testLog.log"
