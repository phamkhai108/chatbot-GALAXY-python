import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Hàm tách các từ thành tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Hàm giảm từ về gốc, chuyển thành lowercase và loại bỏ dấu
def stem(word):
    normalized_word = unidecode(word).lower()
    return stemmer.stem(normalized_word)

# Hàm biểu diễn từ
def bag_of_words(tokenized_sentence, all_words):
    # Stem các từ trong tokenized_sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    # Tạo mảng bằng độ dài all_words và gán là 0.0
    bag = np.zeros(len(all_words), dtype=np.float32)

    # Lặp từng từ trong all_words, so sánh nếu có trong tokenized_sentence thì gán vị trí ndx = 1
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1
    return bag


# tokenized_sentence = ["hello", "world", "hello"]
# all_words = ["hello", "world", "goodbye"]
# [1,1,0]

