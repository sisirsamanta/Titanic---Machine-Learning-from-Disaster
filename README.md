Overview

This project is a machine learning model to predict passenger survival on the Titanic using the Titanic dataset. It preprocesses the data, extracts features, and applies a Gradient Boosting Classifier to predict survival outcomes.

Dataset

The dataset consists of:

train.csv - Training data containing passenger details and survival outcomes.

test.csv - Test data for making survival predictions.

Preprocessing

The preprocessing function:

Cleans passenger names.

Extracts ticket numbers and ticket items.

Drops unnecessary columns (Ticket and PassengerId).

Model

The project uses:

Gradient Boosting Classifier (Primary Model)

Random Forest Classifier (Alternative Model - commented out)

Features Used

Pclass (Passenger class)

Sex (Gender, one-hot encoded)

SibSp (Number of siblings/spouses aboard)

Parch (Number of parents/children aboard)

Installation & Usage

Prerequisites

Ensure you have Python installed. Install required dependencies using:

pip install numpy pandas scikit-learn

Running the Script

Execute the script with:

python script.py

Output

The model generates a submission.csv file containing Passenger IDs and predicted survival (0 for deceased, 1 for survived).

Performance Evaluation

The script calculates and prints the survival rate of men and women for insights into the dataset distribution.

Author

Your Name (or Kaggle Username)

License

MIT License