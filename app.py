import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Input data files are available in the read-only "../data/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def preprocess(df):
    df = df.copy()
    
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
    return df

for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv("./data/train.csv")
preprocessed_train_data = preprocess(train_data)
preprocessed_train_data = preprocessed_train_data.drop(columns=['Ticket', 'PassengerId'])

train_data.head()

test_data = pd.read_csv("./data/test.csv")
preprocessed_test_data = preprocess(test_data)
preprocessed_test_data = preprocessed_test_data.drop(columns=['Ticket', 'PassengerId'])
preprocessed_test_data.head()

women = preprocessed_train_data.loc[preprocessed_train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = preprocessed_train_data.loc[preprocessed_train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)



y = preprocessed_train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(preprocessed_train_data[features])
X_test = pd.get_dummies(preprocessed_test_data[features])

#model = RandomForestClassifier(n_estimators=125,criterion = "entropy", random_state=1,max_features="sqrt")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.25,max_depth=5, random_state=20,loss="log_loss");
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./data/submission.csv', index=False)
print("Your submission was successfully saved!")