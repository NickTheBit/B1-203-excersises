# Session 4 project
## Participants:
- Nikolaos Gkloumpos
- Malthe H. Boelskift
- Louis Ildal
- Guillermo V. Gutierrez-Bea

## Overall description



### About the scripts

## Part 1

To process the data we dimmentionally reduce them both with PCA and LDA to compare the results after classification. 

A scatter plot was generated to help visualize the data.
![Alt text](plots_data/image.png)



## Part 2
Here we aim to make a bayesian classifier that is able to classify data into the three clases 5, 6 and 8. But this time we use dimensinally reduced data from the datasets using two methods (PCA and LDA) and comparing them. The dimentions of both the training set and dataset are reduced to 2. First, we use the training set to find the statistics of the two features of the three classes, we compute the likelihood and prior and finally the posterior probability of each class so we can perform classification taking the highest probability.

```Python
The classifications for the test dataset using lda are [8 6 6 ... 8 8 8].
The classifications for the test dataset using pca are [5 5 8 ... 5 8 8].
The accuracy (lda) is 0.7103399433427762.
The accuracy (pca) is 0.30276203966005666.
```

The generated groups for each classification can be seen bellow:

#### PCA classification
![Alt text](plots_data/PCA.png)

#### LDA classification
![Alt text](plots_data/LDA.png)