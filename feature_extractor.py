#!/usr/bin/env python3
import os
import re
#import string
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
from spellchecker import SpellChecker

# Folder path
path = "C:\\Users\\vstei\\Desktop\\TCC\\Extracted_data2\\"

# Set the directory
os.chdir(path)

# List initialization
user_list = []
full_text = []
stop_words = set(stopwords.words("portuguese"))

var_avg_word_per_sentence = []
var_avg_sentence_len = []
var_avg_word_len = []
var_avg_word_per_post = []
var_avg_post_len = []
var_perc_long_word_in_post = []
var_perc_short_word_in_post = []
var_avg_url_per_post = []
var_avg_lowercase_word_sentence = []
var_avg_uppercase_word_sentence = []
var_perc_lowercase_word_sentence = []
var_perc_uppercase_word_sentence = []
var_avg_sentence_post = []
var_avg_ponctuation_mark = []
var_avg_function_word = []
var_avg_misspelled_word = []
var_avg_adjectives = []
var_avg_nouns = []
var_avg_verbs = []
var_avg_adverb = []
var_avg_conjunction = []
var_avg_article = []
var_avg_pronoun = []


# Function to calculate percentage
def percentage(part, whole):
  percentage = 100 * float(part)/float(whole)
  return str(percentage) + "%"

# Convert percentage into float
def p2f(x):
  return float(x.strip('%'))/100

# Function to calculate average word per sentence
def fun_average_word_per_sentence(full_text):
  for line in full_text:
    sentenced_word = sent_tokenize(line)
    for i in sentenced_word:
      words = i.split()   
      average_word_len = sum(len(word) for word in words) / len(words)
      var_avg_word_per_sentence.append(average_word_len)

# Function to calculate average sentence lenght
def fun_average_sentence_len(full_text):
  for line in full_text:
    sentenced_word = sent_tokenize(line)
    avg_len = sum(len(x.split()) for x in sentenced_word) / len(sentenced_word)
    var_avg_sentence_len.append(avg_len)

# Function to calculate average word lenght
def fun_average_word_len(full_text):
  for line in full_text:
    word_tolkenized = word_tokenize(line)
    average_word = sum(len(word) for word in word_tolkenized) / len(word_tolkenized)
    var_avg_word_len.append(average_word)

# Function to calculate average word per post
def fun_average_word_post(full_text):
  for line in full_text:
    post_list = line.split()
    average_word_len = sum(len(word) for word in post_list) / len(post_list)
    var_avg_word_per_post.append(average_word_len)

# Function to calculate average post lenght
def fun_average_post_len(full_text):
  for line in full_text:
    var_avg_post_len.append(len(line))

# Function to calculate % of words in posts
def fun_perc_words_posts(full_text):
  for line in full_text:
    long_word_count = 0
    short_word_count = 0
    total_word_count = 0
    word_in_post = word_tokenize(line)
    total_word_count = len(word_in_post)
    for word in word_in_post:
      if len(word) > 7:
        long_word_count += 1
      else:
        short_word_count += 1
    var_perc_long_word_in_post.append(p2f(percentage(long_word_count, total_word_count)))
    var_perc_short_word_in_post.append(p2f(percentage(short_word_count, total_word_count)))

# Function to calculate average URL per post
def fun_average_url_per_post(full_text):
  regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
  for line in full_text:
    url = re.findall(regex,line)
    url_count = 0
    for i in url:
      url_count += 1
    var_avg_url_per_post.append(url_count)

# Function to calculate average/percentage lower/uppercase word in sentence
def fun_average_lower_upper_word_sentence(full_text):
  for line in full_text:
    lower_word = 0
    upper_word = 0
    total_word = 0
    if line.islower():
      lower_word += 1
      total_word += 1
    else:
      upper_word += 1
      total_word += 1

    var_avg_lowercase_word_sentence.append(lower_word)
    var_avg_uppercase_word_sentence.append(upper_word)

    var_perc_lowercase_word_sentence.append(percentage(lower_word, total_word))
    var_perc_uppercase_word_sentence.append(percentage(upper_word, total_word))

# Function to calculate the average sentence per post
def fun_average_sentence_per_post(full_text):
  for line in full_text:
    sentenced_word = sent_tokenize(line)
    sentence_count = 0
    for i in sentenced_word:
      sentence_count += 1
    var_avg_sentence_post.append(sentence_count)

# Function to calculate the average ponctuaction marks in text
def fun_average_ponctuation_mark(full_text):
  for line in full_text:
    ponctuation_count = 0
    for i in range (0, len (line)):   
      if line[i] in ('!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?"):  
        ponctuation_count = ponctuation_count + 1
    var_avg_ponctuation_mark.append(ponctuation_count)

# Function to calculate the average of function words in text
def fun_average_function_word(full_text):
  for line in full_text:
    adjectives = 0
    nouns = 0
    verbs = 0
    adverb = 0
    conjunction = 0
    article = 0
    pronoun = 0
    sentenced_word = sent_tokenize(line)
    for sentence in sentenced_word:
      sentence_tokens = word_tokenize(sentence)
      tagged_sentence_tokens = nltk.pos_tag(sentence_tokens)
      #var_avg_function_word.append(Counter([j for i,j in pos_tag(word_tokenize(ii))]))
      
      adjectives = [word for (word,pos) in tagged_sentence_tokens if pos == 'JJ' or pos == 'JJR' or pos == 'JJS']
      nouns = [word for (word,pos) in tagged_sentence_tokens if pos=='NN' or pos=='NNS']
      verbs = [word for (word,pos) in tagged_sentence_tokens if pos in ['VB', 'VBD','VBG','VBN','VBP','VBZ']]
      adverb = [word for (word,pos) in tagged_sentence_tokens if pos == 'RB' or pos == 'RBR' or pos == 'RBS']
      conjunction = [word for (word,pos) in tagged_sentence_tokens if pos == 'CC']
      article = [word for (word,pos) in tagged_sentence_tokens if pos == 'DT']
      pronoun = [word for (word,pos) in tagged_sentence_tokens if pos == 'PRP' or pos == 'PRP$']      
      
    var_avg_adjectives.append(len(adjectives))
    var_avg_nouns.append(len(nouns))
    var_avg_verbs.append(len(verbs))
    var_avg_adverb.append(len(adverb))
    var_avg_conjunction.append(len(conjunction))
    var_avg_article.append(len(article))
    var_avg_pronoun.append(len(pronoun))  
      
# Function to calculate the average of misspelled words
def fun_average_misspelled_word(full_text):
  for line in full_text:
    word_list = line.split()
    spell = SpellChecker(language='pt')
    misspelled_count = len(list(spell.unknown(word_list)))
    var_avg_misspelled_word.append(misspelled_count)

# Remove unecessary letters from text 
def fun_remove_unecessary_words(x_word):
    x_word_0 = "\n".join(map(str, x_word))
    x_word_1 = x_word_0.replace('\"','')
    x_word_2 = x_word_1.replace('(','')
    x_word_3 = x_word_2.replace(')','')
    x_word_4 = x_word_3.replace('Counter{','')
    x_word_5 = x_word_4.replace('}','')
    return x_word_5
    
# Function to read all files in path
def read_text_file(file_path):
  with open(file_path, 'r', encoding='utf8') as file:
    lines_in_file = file.readlines()

    for l in lines_in_file:
      user_temp,text_temp = l.split(",",1)
      user_list.append(user_temp)
      full_text.append(text_temp)

    # Call average word per sentence function
    fun_average_word_per_sentence(full_text)

    # Call average sentence lenght function
    fun_average_sentence_len(full_text)

    # Call average word lenght function
    fun_average_word_len(full_text)

    # Call average word per post function
    fun_average_word_post(full_text)

    # Call average post lenght function
    fun_average_post_len(full_text)

    # Call % of words in post function
    fun_perc_words_posts(full_text)

    # Call average URL per post function
    fun_average_url_per_post(full_text)

    # Call average/percentage lower/uppercase word in sentence function
    fun_average_lower_upper_word_sentence(full_text)

    # Call average sentence per post function
    fun_average_sentence_per_post(full_text)

    # Call average ponctuaction marks in text function
    fun_average_ponctuation_mark(full_text)

    # Call average of function words in text fuction
    fun_average_function_word(full_text)

    # Call average of misspelled words function
    #fun_average_misspelled_word(full_text)

# Iterate over all the files in the directory
for file in os.listdir():
  # Check whether file is in text format or not
  if file.endswith(".txt"):
    file_path = f"{path}/{file}"

    # Call read text file function
    read_text_file(file_path)

# Concatenate list
#row0 = list(zip(user_list, var_avg_word_per_sentence, var_avg_sentence_len, var_avg_word_len, var_avg_word_per_post, var_avg_post_len, var_perc_long_word_in_post, var_perc_short_word_in_post, var_avg_url_per_post, var_avg_lowercase_word_sentence, var_avg_uppercase_word_sentence, var_avg_sentence_post, var_avg_ponctuation_mark, var_avg_misspelled_word, var_avg_adjectives, var_avg_nouns, var_avg_verbs, var_avg_adverb, var_avg_conjunction, var_avg_article, var_avg_pronoun))
row0 = list(zip(user_list, var_avg_word_per_sentence, var_avg_sentence_len, var_avg_word_len, var_avg_word_per_post, var_avg_post_len, var_perc_long_word_in_post, var_perc_short_word_in_post, var_avg_url_per_post, var_avg_lowercase_word_sentence, var_avg_uppercase_word_sentence, var_avg_sentence_post, var_avg_ponctuation_mark, var_avg_adjectives, var_avg_nouns, var_avg_verbs, var_avg_adverb, var_avg_conjunction, var_avg_article, var_avg_pronoun))
#csv_final_list_01 = fun_remove_unecessary_words(row0)
#csv_final_list_02 = fun_remove_unecessary_words(row0_2)


#histogram = Counter(row0_2)
#results = [d for d in row0_2 if histogram[d] >= 9]

# Using Pandas to create the CSV file
df = pd.DataFrame(row0)
#df.columns = ['Username', 'Average word per sentence', 'Average sentence lenght', 'Average word lenght', 'Average word per post', 'Average post lenght', 'Average longwords in post', 'Average shortwords in post', 'Average URL per post', 'Average lowercase starting sentences', 'Averave uppercase starting sentences', 'Average sentence per post', 'Average ponctuation mark per post', 'Average misspelled words per post', 'adjectives', 'nouns', 'verbs', 'adverb', 'conjunction', 'article', 'pronoun']
df.columns = ['Username', 'Average word per sentence', 'Average sentence lenght', 'Average word lenght', 'Average word per post', 'Average post lenght', 'Average longwords in post', 'Average shortwords in post', 'Average URL per post', 'Average lowercase starting sentences', 'Averave uppercase starting sentences', 'Average sentence per post', 'Average ponctuation mark per post', 'adjectives', 'nouns', 'verbs', 'adverb', 'conjunction', 'article', 'pronoun']

is_multi = df["Username"].value_counts() >= 150
filtered = df[df["Username"].isin(is_multi[is_multi].index)]

#if df['Username'] == '(Brodie' or '(SkyfolksV.2' or '(Discórdia' or '(ViniZx ¥' or '(Xandão do STF' or '(lnlmdk' or '(Senhor White' or '(Аиоnт' or '(Mohamed Salah' or '(mainframe32':
#  is_multi = df["Username"].value_counts() >= 90
#  filtered = df[df["Username"].isin(is_multi[is_multi].index)]

#filtered.drop(['Username'], axis=1)

#print(filtered['Username'])
#print('\n\n\n')
#print(df)

# Priting training/testing dataset CSV
#df.to_csv('C:\\Users\\vstei\\Desktop\\TCC\\Profile\\profile_csv_file_12_test.csv')
filtered.to_csv('C:\\Users\\vstei\\Desktop\\TCC\\Profile\\profile_csv_file_7users_150posts.csv')

print("Success!")
