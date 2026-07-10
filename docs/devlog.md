*Day 1:*
Datasets downloaded and preprocessed.
elo dataset: https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings/data
results dataset: https://github.com/martj42/international_results/blob/master/results.csv

*preprocessing steps:*
1. Load the datasets into pandas DataFrames.
2. Define input and output paths
3. Define team map for standardizing team names across datasets and creating function for the standardization.
4. Creating result function
5.Load the datasets into pandas DataFrames.
6. Standardize Dates
7. Drop null values and duplicates
8. Filter the datasets to only include relevant data from 2016 onwards
9. Standardize team names using the team map and the standardization function.
10. Removing matches without elo ratings because they are not useful for the model. (some matches are not international football matches and this is a fifa result predictor)
11. Save the preprocessed datasets to CSV files for future use.

*feature engineering steps:*
1. function to get latest elo rating for each team before the match date
2. funtion to add the features: home_elo, away_elo, elo_diff
3. Save feature engineered datasets to CSV files for future use.


*model training steps:*
1. Load the feature engineered datasets into pandas DataFrames.
2. Split the data into training and testing sets.
3. Train a logistic regression model using the training set.
4. Evaluate the model's performance on the testing set using accuracy, confusion matrix, and classification report.
5. Save the trained model and evaluation metrics to files for future use.