import tokenizer as text_rank_tokenizer
import ranker as text_rank_ranker
import numpy as np

stop_words = open("./text_rank/stopwords.txt",'r',encoding="utf-8").read()
word_endings = open("./text_rank/word_endings.txt",'r',encoding='utf-8').read() 
kriyapads = open("./text_rank/minimal_kriyapad.txt",'r',encoding="utf-8").read().split("\n")
samyojaks = open("./text_rank/samyojak.txt",'r',encoding="utf-8").read().split("\n")
valid_chars = "./text_rank/valid_chars.json"

def get_summary_from_text(text,force_use_purnabiram_model=False):
    global stop_words, word_endings, kriyapads, samyojaks
    # 
    # Reading text files (sample text file, word endings file and stopwords file)
    #
    # text = open(file_path,'r',encoding="utf-8").read()
    #
    is_complete_sentence = True
    # if "।" not in text:
    purnabiram_count = text.count("।") 
    if not force_use_purnabiram_model:
        if purnabiram_count*100 < len(text):
            is_complete_sentence = False
    else:
        is_complete_sentence = False
    # print(is_complete_sentence)   

    valid_characters = text_rank_tokenizer.get_valid_chars(valid_chars)
    # print(stop_words.split("\n"))
    # print(text)
    #
    # Remove useless characters from the sentence 
    # 
      
    if not is_complete_sentence:
        text = text_rank_tokenizer.add_purnabiram(text,kriyapads,samyojaks)
    
    #
    # Split the sentence into array of words and patagraph in its array. (as Array of Array of the words)
    #
    sentences = text_rank_tokenizer.get_sentences_as_arr(text)
    # print(sentences)

    text = text_rank_tokenizer.remove_useless_characters(text,valid_characters)


    sentences = text_rank_tokenizer.remove_repeating_sentences(sentences)
    
    if len(sentences) == 0:
        return "It is not a valid text. Please try again with a valid text."
    elif len(sentences) == 1:
        return sentences
    
    # print(sentences)
    words_arr = text_rank_tokenizer.get_words_as_arr(sentences)    
    #
    # Remove the stop words from the array
    #
    words_arr = text_rank_tokenizer.remove_stop_words_and_filter_word_arr(words_arr,word_endings, stop_words)
    # print(words_arr)
    
    #
    # remove empty sentences and lone word sentences and update sentences accordingly
    #    
    sentences, words_arr = text_rank_tokenizer.remove_empty_sentences(sentences, words_arr)
    #
    # Tokenize the words and sentences into numbers
    # 
    tokens, token_dict = text_rank_tokenizer.tokenize(words_arr)
    # 
    # Create a association matrix
    # 
    association_matrix, counter_vector = text_rank_ranker.create_association_matrix(tokens,No_of_unique_chars= len(token_dict))
    # 
    # Calculate influence of each word on the paragraph
    # 
    word_influence_vector = text_rank_ranker.calculate_word_ranks(association_matrix, counter_vector)
    # 
    # Based in the word importance ranking, calculate teh sentence importance ranking.
    # 
    sentence_influence = text_rank_ranker.calculate_sentence_influence(tokens,word_influence_vector)
    
    # 
    # Get first n sentences from the given text as summarized text.
    # 
    
    # print(sentence_influence)
    summary_sentences = text_rank_ranker.get_n_influencial_sentence(sentences,sentence_influence,n=np.ceil(len(sentences)*0.33))

    #
    # Combine all sentences as a single paragraph
    #
    summarized_text = text_rank_ranker.get_summarized_text(summary_sentences)
    
    # with open(outputfile, 'w',encoding="utf-8") as f:
    #     f.write(summarized_text)
    return summarized_text



# with open(f"D:\\speechrecog\\nepali-asrs\\transcripts\\anushasan.txt",'r',encoding="utf-8") as f:
#     article_text=f.read()

# get_summary_from_text(article_text)