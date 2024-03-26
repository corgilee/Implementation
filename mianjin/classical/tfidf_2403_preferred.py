import math
from collections import Counter

class TFIDFVectorizer:
    def __init__(self):
        self.documents = []
        self.idf_values = {}
    
    def add_document(self, document):
        """Add a document to the corpus."""
        self.documents.append(document)
    
    def calculate_tf(self, document):
        """Calculate term frequency for a document."""
        tf = Counter(document.split())
        total_terms = len(document.split())
        for term in tf:
            tf[term] = tf[term] / total_terms
        return tf
    
    def calculate_idf(self):
        """Calculate inverse document frequency for the corpus."""
        total_documents = len(self.documents)
        term_in_docs = Counter(word for document in self.documents for word in set(document.split()))
        
        for term, count in term_in_docs.items():
            self.idf_values[term] = math.log(total_documents / count)
    
    def fit(self):
        """Calculate IDF values based on the current state of the corpus."""
        self.calculate_idf()
    
    def transform(self, document):
        """Calculate TF-IDF for a given document."""
        tf = self.calculate_tf(document)
        tfidf = {}
        for word, tf_value in tf.items():
            tfidf[word] = tf_value * self.idf_values.get(word, 0)
        return tfidf

# Example usage
if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat sat in the hat",
    ]
    
    vectorizer = TFIDFVectorizer()
    for doc in documents:
        vectorizer.add_document(doc)
    
    vectorizer.fit()
    
    # Transform a document to its TF-IDF representation
    tfidf_representation = vectorizer.transform("the cat sat on the mat")
    print(tfidf_representation)
