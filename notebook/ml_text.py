import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Punctuations
import string
import pandas as pd


# Regular Expressions
import re
# Import PorterStemmer from NLTK Library
from nltk.stem.porter import PorterStemmer
# Lemmatization
from nltk.stem import WordNetLemmatizer
# Tokenization
from nltk.tokenize import word_tokenize
# Imporr Ohe 
from sklearn.preprocessing import OneHotEncoder

# text lowercase
df['review'] = df['review'].str.lower()
# define punctuation
punctuation = string.punctuation
translate_table=str.maketrans('', '', string.punctuation)

# Apply the function to the 'text' column
df['review'] = df['review'].str.translate(translate_table)

# Remove Stopwords
from nltk.corpus import stopwords 
stop_words = stopwords.words('english')


def remove_stopwords(text):
  res=[word for word in text.split() if word not in stop_words]
  return " ".join(res)

df['review']=df['review'].apply(remove_stopwords)

### Tokenization (optional)
from nltk.tokenize import word_tokenize
# Apply word_tokenize
df['review_word_token'] = df['review'].apply(word_tokenize)



### TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])

# Convert the TF-IDF matrix to a dense array for easier manipulation (optional)
tfidf_matrix_dense = tfidf_matrix.toarray()

# Get the feature names (words) from the vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()

### DistillBert
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import BertTokenizer
import torch

# Load pretrained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.eval()  # Put the model in evaluation mode

def text_to_embedding(text, tokenizer, model):
    # Tokenize and prepare the inputs
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=100)
    # return_tensors="pt" means return pytorch tensors
    with torch.no_grad():
        '''
        这里的input 其实是个dictionary，包含 ['input_ids', 'token_type_ids', 'attention_mask']
        '''
        outputs = model(**inputs) 
        

    # Extract embeddings from the last hidden state
    last_hidden_states = outputs.last_hidden_state
    # Use the output of the `[CLS]` token (first token) as the sequence representation
    embeddings = last_hidden_states[:, 0, :].squeeze().numpy()
    return embeddings

df['embeddings']=df['review'].apply(lambda x: text_to_embedding(x, tokenizer, model))

embeddings_array = np.stack(df['embeddings'].values) #combining a sequence of arrays along a new axis
df_embedding=pd.DataFrame(embeddings_array)

