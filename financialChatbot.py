## FINANCIAL CHATBOT ##
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class FinancialChatbot:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.questions = self.df['Question']
        self.answers = self.df['Answer']
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
        self.X = self.vectorizer.fit_transform([self.preprocess(q) for q in self.questions])

    def preprocess(self, user_input): # process user input and remove stopwords
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokens = nltk.word_tokenize(user_input.lower())
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
        
        return ' '.join(stemmed_tokens)
        
        
    def get_response(self, user_input):
        # use cosine similarity to find the closest question
        processed_text = self.preprocess(user_input)
        # print(processed_text)
        
        vectorized_text = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(vectorized_text, self.X)
        # print(similarities)
        
        max_similarity = np.max(similarities)
        if max_similarity > .6:
            closest_question = np.argmax(similarities)
            # print(max_similarity)
            print(self.answers[closest_question])        
            return self.answers[closest_question]
        else:
            return "I'm sorry, I don't understand that question. Can you please ask something else?"