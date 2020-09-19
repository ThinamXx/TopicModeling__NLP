# **Natural Language Processing**

**Fastai Library or API**
- [Fast.ai](https://www.fast.ai/about/) is the first deep learning library to provide a single consistent interface to all the most commonly used deep learning applications for vision, text, tabular data, time series, and collaborative filtering.
- [Fast.ai](https://www.fast.ai/about/) is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches.

### [**01 Topic Modeling with SVD and NMF**](https://github.com/ThinamXx/NaturalLanguageProcessing_NLP/blob/master/Topic%20Modeling%20with%20SVD%20and%20NMF.ipynb)

**Objectives and Overview**
- I have prepared a Topic Modeling with Singular Value Decomposition (SVD) and NonNegative Factorization (NMF) and Topic Frequency Inverse Document Frequency (TFIDF). I have also performed some basic Exploratory Data Analysis such as Visualization and Processing the Data. I have tried to explain everything on this Project with proper documentation. I have also covered the basic techniques such as Stopwords, Stemming and Lemmatization in this Project. 

**Libraries and Dependencies**

```javascript
import nltk
import numpy as np  
from sklearn import decomposition
from scipy import linalg
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
```

**Singular Value Decomposition (SVD)**
- The words that appear most frequently in one topic would appear less frequently in the other, otherwise that word wouldn't make a good choice to separate out the two topics. Therefore, the topics are Orthogonal. The SVD algorithm factorizes a matrix into one matrix with orthogonal columns and one with orthogonal rows along with diagonal matrix which contains the relative importance of each factor. SVD is an exact decomposition since the matrices it creates are big enough to fully cover the original matrix. Basic implementations of SVD are: 
  - Semantic Analysis.
  - Collaborative Filtering or Recommendations System.
  - Data Compression.
  - Principal Component Analysis (PCA).
  
- Snapshot of the Diagonal Matrix obtained using SVD : 

![Image](https://github.com/ThinamXx/66DaysofData__NLP/blob/master/Images/O1.PNG)

**NonNegative Matrix Factorization (NMF)**
- Non Negative Matrix Factorization (NMF) is a factorization or constrain of non negative dataset. NMF is non exact factorization that factors into one short positive matrix. Basic implementations of NMF are:
  - Face Decompositions.
  - Collaborative Filtering or Movie Recommendations.
  - Audio Source Separation.

**Topic Frequency Inverse Document Frequency (TFIDF)**
- TFIDF is a way to normalize the term counts by taking into account how often they appear in a document and how long the document is and how common or rare the document is.

- Snapshot of the result obtained from NMF and TFIDF :

![Image](https://github.com/ThinamXx/66DaysofData__NLP/blob/master/Images/Day%2015%20a.PNG)

**Topics Obtained**
- The 5 topics obtained from the Implementation of NMF and TFIDF is:

```javascript
['jpeg image gif file color images format quality version files',
 'edu graphics pub mail 128 ray ftp send 3d com',
 'space launch satellite nasa commercial satellites year market data earth',
 'jesus god people matthew atheists does atheism said just believe',
 'image data available software processing ftp edu analysis images display']
```
