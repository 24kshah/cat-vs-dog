import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

image_folder = r"C:\Users\24ksh\OneDrive\Desktop\cat vs dog\data\raw\train\train"

print("Files in folder:", os.listdir(image_folder)[:10])  # print first 10 filenames

filenames = os.listdir(image_folder)
df = pd.DataFrame({'filename': filenames})
df['label'] = df['filename'].apply(lambda x: 'cat' if 'cat' in x else 'dog')

print(df.head())
print(df['label'].value_counts())

sns.countplot(x='label', data=df)
plt.title('Count of Cats vs Dogs Images')
plt.show()
from sklearn.preprocessing import LabelEncoder

df['label_encoded'] = df['label'].map({'cat': 0, 'dog': 1})
print(df.head())

