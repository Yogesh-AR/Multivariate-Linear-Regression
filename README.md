# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:

## Step1
<br>
import pandas as pd.
## Step2
<br>
Read the csv file.
## Step3
<br>
Get the value of X and y variables
## Step4
<br>
Create the linear regression model and fit.
## Step5
<br>
Perdict the Train data and testing data
## Program:

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X = data
y = target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

print('Coefficients: \n', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()
```
## Output:
### Insert your output
<img width="1257" height="587" alt="image" src="https://github.com/user-attachments/assets/9ebd1189-2413-4e91-84b1-ee5337ac3b5a" />
<img width="1115" height="704" alt="image" src="https://github.com/user-attachments/assets/16f0545a-0af6-4da7-ab78-9303af7c80e5" />
## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
