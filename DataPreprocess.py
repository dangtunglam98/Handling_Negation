import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from nltk.corpus import stopwords
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('stopwords')
NEGATE = {
    "n't",
    "aint",
    "arent",
    "cannot",
    "cant",
    "couldnt",
    "darent",
    "didnt",
    "doesnt",
    "ain't",
    "aren't",
    "can't",
    "couldn't",
    "daren't",
    "didn't",
    "doesn't",
    "dont",
    "hadnt",
    "hasnt",
    "havent",
    "isnt",
    "mightnt",
    "mustnt",
    "neither",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "mightn't",
    "mustn't",
    "neednt",
    "needn't",
    "never",
    "none",
    "nope",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "oughtnt",
    "shant",
    "shouldnt",
    "uhuh",
    "wasnt",
    "werent",
    "oughtn't",
    "shan't",
    "shouldn't",
    "uh-uh",
    "wasn't",
    "weren't",
    "without",
    "wont",
    "wouldnt",
    "won't",
    "wouldn't",
    "rarely",
    "seldom",
    "despite",
}

data_frame = pd.read_json(r'reviews_Automotive_5.json', lines = True)

data_frame.head()

data_frame = data_frame.dropna()
data_frame = data_frame.drop(['asin', 'helpful', 'reviewTime', 'reviewerID', 'reviewerName', 'summary', 'unixReviewTime'], axis = 1)

def negated(input_words, include_nt=True):
    """
    Determine if input contains negation words
    """
    neg_words = NEGATE
    if any(word.lower() in neg_words for word in input_words):
        return True
    if include_nt:
        if any("n't" in word.lower() for word in input_words):
            return True
#     for first, second in pairwise(input_words):
#         if second.lower() == "least" and first.lower() != 'at':
#             return True
    return False

def negated_word(word, include_nt=True):
    neg_words = NEGATE
    if word in neg_words:
        return True
    if include_nt:
        if word == "n't":
            return True
        
    return False

def get_negation_statistic(data_frame):
    #Negation Counter
    total_negation_count = 0
    total_negation_sentence_count = 0
    
    #Total sentence counter
    total_sentence_count = 0
    
    #Sentiment Counter
    total_sentiment_count = 0
    total_negation_sentiment_count = 0
    
    reviewList = data_frame['reviewText'].tolist()
    for sentence in reviewList:
        sentence_tokenizer = word_tokenize(sentence)
        review_negation_count = 0
        review_sentiment_count = 0
        for word in sentence_tokenizer:
            if negated_word(word):
                review_negation_count += 1
            if word == '.': #at the end of the sentence
                total_sentence_count += 1
                if review_negation_count == 0: #There's no negation word
                    pass
                else: #There's a negation
                    if review_negation_count > 1:
                        print(sentence)
                    total_negation_count += review_negation_count
                    total_negation_sentence_count += 1
                    review_negation_count = 0
    return total_negation_count, total_negation_sentence_count, total_sentence_count, total_negation_count/total_negation_sentence_count
    
stop_words = set(stopwords.words('english'))
def get_sentiment(sentence):
    sentiment_list = []
    sentiment_pos_scores = []
    sentiment_neg_scores = []
    sentence_token = nltk.word_tokenize(sentence)
    for wordIndex in range(len(sentence_token)):
        word = sentence_token[wordIndex].lower()
        if word in stop_words or word in NEGATE:
            pass
        else:
            token = lesk(sentence_token, word)
            if token is not None:
                token_name = token.name()
                pos_score = swn.senti_synset(token_name).pos_score()
                neg_score = swn.senti_synset(token_name).neg_score()
                if pos_score > 0.125 or neg_score > 0.125:
                    sentiment_list.append(wordIndex)
                    sentiment_pos_scores.append[pos_score]
                    sentiment_neg_scores.append[neg_score]
    return sentiment_list, sentiment_pos_scores, sentiment_neg_scores


def get_negation(sentence):
    negation_list = []
    sentence_token = nltk.word_tokenize(sentence)
    for wordIndex in range(len(sentence_token)):
        word = sentence_token[wordIndex].lower()
        if negated_word(word):
            negation_list.append(wordIndex)
    return negation_list, len(negation_list)
    

def preprocess_sentence(sentence):
    pass
