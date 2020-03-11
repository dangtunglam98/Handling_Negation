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
    max_negation_count = 0
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
                    if review_negation_count > max_negation_count:
                        print(review_negation_count)
                        max_negation_count = review_negation_count
                    total_negation_count += review_negation_count
                    total_negation_sentence_count += 1
                    review_negation_count = 0
    return total_negation_count, total_negation_sentence_count, total_sentence_count, total_negation_count/total_negation_sentence_count, max_negation_count
    
stop_words = set(stopwords.words('english'))
def get_sentiment(sentence_token):
    positive_index = []
    negative_index = []
#     sentence_token = nltk.word_tokenize(sentence)
    for wordIndex in range(len(sentence_token)):
        word = sentence_token[wordIndex].lower()
        if word in stop_words or word in NEGATE:
            pass
        else:
            if word in NEGATIVE:
                  negative_index.append(wordIndex)  
            elif word in POSITIVE:
                positive_index.append(wordIndex)
    return positive_index, negative_index


def get_negation(sentence_token):
    negation_list = []
    pos_after_negation = []
    for wordIndex in range(len(sentence_token)):
        word = sentence_token[wordIndex].lower()
        if negated_word(word):
            negation_list.append(wordIndex)
            
    return negation_list, len(negation_list)
    
def is_closest_sentiment_positive(value, pos_list, neg_list):
    smallest = 1000000
    positive = True
    for item in pos_list:
        if item > value and item < smallest:
            smallest = item
    for item in neg_list:
        if item > value and item < smallest:
            positive = False
    if smallest == 1000000:
        positive = None
    return positive

def calculate_negate_sentiment_score(value, pos_list, neg_list):
    is_positive = True
    closest_index = 10000000
    for item in pos_list:
        if item > value and item < closest_index:
            closest_index = item
    for item in neg_list:
        if item > value and item < closest_index:
            closest_index = item
            is_positive = False
    if closest_index == 10000000:
        return 0
    if is_positive == False:
        score = 1/(closest_index - value)
    else:
        score = -1/(closest_index - value)
    
    return score


def preprocess_sentence(sentence):
    vector = []
    sentence_token = nltk.word_tokenize(sentence)
    pos_index , neg_index = get_sentiment(sentence_token)
    negate_index, num_negate = get_negation(sentence_token)
    pos_score = len(pos_index)
    neg_score = len(neg_index)
    num_sent = pos_score + neg_score
    if num_sent == 0:
        return None
    else:
        vector.append(num_negate)
        vector.append(num_sent)
        vector.append(pos_score)
        vector.append(neg_score)
        true_pos = pos_score
        true_neg = neg_score
        negate_score = 0
        for negateIndex in negate_index:
            closest_is_positive = is_closest_sentiment_positive(negateIndex, pos_index, neg_index)
            if closest_is_positive == True:
                true_pos -= 1
            elif closest_is_positive == False:
                true_neg -= 1
            else:
                pass
            negate_score += calculate_negate_sentiment_score(negateIndex, pos_index, neg_index)
        accumulative_score = true_pos + negate_score - true_neg
        vector.append(accumulative_score)
    
        return vector

def preprocess_review(review):
    review_vector = [0,0,0,0,0]
    review_token = sent_tokenize(review)
    for sentence in review_token:
        sentence_vector = preprocess_sentence(sentence)
        if sentence_vector == None:
            pass
        else:
            review_vector = [sum(i) for i in zip(sentence_vector, review_vector)] 
    return review_vector

def preprocess_dataframe(list_review):
    preprocessed = []
    for review in list_review:
        preprocessed.append(preprocess_review(review))
    
    preprocessed_df = pd.DataFrame(preprocessed, columns=["negation_count", "sentiment_count", "positive_count",
                                                         "negative_count", "accumulative_score"])
    return preprocessed_df
    
review = data_frame['reviewText'].tolist()
preprocess_dataframe(review)