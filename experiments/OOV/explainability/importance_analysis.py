import os
from collections import Counter


# load the csv
with open("outputs/oov/ig/most_important_words_pred_class_all_preds_v2.csv", "r") as f:
    data = f.readlines()

# Create a dictionary with the intent as key and the important words as value
def intent_to_words(data):
    intent2important_words = {}
    for line in data:
        file_id, transcript, scenario, action, important_word = line.split(",")
        important_word = important_word.strip()
        intent = scenario + "_" + action
        if intent not in intent2important_words:
            intent2important_words[intent] = [important_word]
        else:
            intent2important_words[intent].append(important_word)
    return intent2important_words


# Create a dictionary with the intent as key and a dictionary of the word counts as value
def intent_to_count(intent2important_words):
    intent2count = {}
    for key, words in intent2important_words.items():
        # Convert all words to lowercase
        words = [word.lower() for word in words]
        
        # Count the occurrences of each word while preserving order
        word_counts = dict(Counter(words))
        # order the dictionary by value
        word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
        
        # Add the result to the new dictionary
        intent2count[key] = word_counts
    return intent2count


intent2important_words = intent_to_words(data)
intent2count = intent_to_count(intent2important_words)

# get top n words
n = 100
top_n_values = {
    outer_key: sorted(inner_dict.items(), key=lambda item: item[1], reverse=True)[:n]
    for outer_key, inner_dict in intent2count.items()
}

# sort the dict by key alphabetically
top_n_values = dict(sorted(top_n_values.items(), key=lambda item: item[0]))

for key, value in top_n_values.items():
    print(key,  value)
