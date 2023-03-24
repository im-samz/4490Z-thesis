# Import Libraries
import torch
from transformers import BertTokenizer, BertModel

import numpy as np
from scipy.spatial.distance import cosine
import csv

# Define the models we will use
checkpoint1 = 'bert-base-uncased'
checkpoint2 = 'Charangan/MedBERT'
checkpoint3 = "mental/mental-bert-base-uncased"

#! Access token is required to use "AIMH/mental-bert-large-uncased". 
# https://huggingface.co/docs/hub/security-tokens
access_token = "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"

# Select the model to run for the experiment
checkpoint = checkpoint2
checkpoint_name = checkpoint.replace('/','_') # For use when saving the .csv filename

# Check for errors
error_count = 0

# Option 1: Concatenate the last 4 layers: WV length of 4*768=3072
def cat_last_four(token_embeddings, index):
    
    # Stores the token vectors, with shape [n x 3,072], with n being number of tokens
    token_vecs_cat = []

    # For each token in the sentence...
    for token in token_embeddings:
        
        # `token` is a [n x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    #! print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

    return token_vecs_cat[index]    # return just the vector for the target word

# Option 2: Sum the last four layers: WV length of 768, but magnitude is higher
def sum_last_four(token_embeddings, index):
    
    # Stores the token vectors, with shape [n x 768]
    token_vecs_sum = []

    # For each token in the sentence...
    for token in token_embeddings:

        # `token` is a [n x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    #! print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))

    return token_vecs_sum[index]    # return just the vector for the target word

# Option 3: Get the second last hidden layer: WV length of 768
def get_second_last(token_embeddings, index):

    # Get the second last layer
    token_vecs_second_last = token_embeddings[index][-2]    # return just the vector for the target word

    return token_vecs_second_last

def get_word_vectors(input_text, target_word):
    
    # Add the special tokens.
    marked_text = "[CLS] " + input_text + " [SEP]"
       
    #* Step : Tokenize the marked input text
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(checkpoint, use_auth_token="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")

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Generate the indexes for each token
    indexes = list(range(len(indexed_tokens)))

    '''
    # Display the words with their indeces.
    for tup in zip(indexes, tokenized_text, indexed_tokens):
        print('{:<3} {:<12} {:>6,}'.format(tup[0], tup[1], tup[2]))
    '''
        
    # Mark each of the tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    #* Step : pass the input text into the BERT Model
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(checkpoint, output_hidden_states = True, use_auth_token="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")

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced from all 12 layers. 
    with torch.no_grad():

        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. 
        # See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    #* Step : Adjust the dimensionality of the hidden layers to be useful
    # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings.size() #[13,1,22,768]

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings.size() #[13,22,768]

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    token_embeddings.size() #[22,13,768]

    #* Step : Now it is time to set up the experiment!
    # Get the index of the target word to get a specific word vector
    target_word_index = 0
    try:
        target_word_index = tokenized_text.index(target_word)
    except:
        print("Error: decomposed target word: %s" % target_word)
        error_count += 1
    
    clf_wv = cat_last_four(token_embeddings, target_word_index)
    slf_wv = sum_last_four(token_embeddings, target_word_index)
    gsl_wv = get_second_last(token_embeddings, target_word_index)

    word_vectors = [input_text, target_word, checkpoint, [clf_wv, slf_wv, gsl_wv]] #! currently only looks at the first BERT Model.
    
    return word_vectors

def cosine_similarity(word_vector_1, word_vector_2):

    cosine_diff_1 = 1 - cosine(word_vector_1[0], word_vector_2[0]) # Concatenate Last Four
    cosine_diff_2 = 1 - cosine(word_vector_1[1], word_vector_2[1]) # Sum Last Four
    cosine_diff_3 = 1 - cosine(word_vector_1[2], word_vector_2[2]) # Get Second Last

    cosine_statistics = [cosine_diff_1, cosine_diff_2, cosine_diff_3]

    return cosine_statistics

def dot_product(word_vector_1, word_vector_2):

    dot_product_1 = np.tensordot(word_vector_1[0], word_vector_2[0], axes=1)
    dot_product_2 = np.tensordot(word_vector_1[1], word_vector_2[1], axes=1)
    dot_product_3 = np.tensordot(word_vector_1[2], word_vector_2[2], axes=1)

    dot_product_statistics = [dot_product_1, dot_product_2, dot_product_3]

    return dot_product_statistics

# Takes in a pair of words, finds the word vector for the first occurence of each word, and performs cosine and dot products.
# Output: array of [2 target words, 3 cosine statistics, 3 dot products]
def process_word_pair(target_words, word_vector_array):

    # Initialize a variable to hold the vector
    word_vector_1 = []
    word_vector_2 = []
    
    # Get the word vector of the first word from the array of word vectors
    for entry in word_vector_array:
        if entry[1] == target_words[0]:
            word_vector_1 = entry[3]
        
        elif entry[1] == target_words[1]:
            word_vector_2 = entry[3]
    
    if not (word_vector_1 or word_vector_2):
        print("Error: target words from word-pair were not inputted.")
        return

    cosine_statistics = cosine_similarity(word_vector_1, word_vector_2)
    dot_product_statistics = dot_product(word_vector_1, word_vector_2)

    statistics = [target_words, cosine_statistics, dot_product_statistics]

    return statistics

def print_single_word_statistics(statistics_for_single_word):
    print("PRINTING STATISTICS FOR MULTI-CONTEXTS OF: ", statistics_for_single_word[0][0][0])
    print("-"*80)

    for statistic in statistics_for_single_word:
        print("COSINE SIMILARITY: ", statistic[1])
        print("DOT PRODUCT: ", (float(statistic[2][0]), float(statistic[2][1]), float(statistic[2][2])))
        print()
    
    print()

# Experiment output is a three dimensional array: word-pair x type of statistic x specific statistic
def print_statistics(experiment_output):

    print("PRINTING STATISTICS FOR WORD PAIRS")
    print("-"*80, '\n')

    for statistic in experiment_output:
        print("WORD PAIR: ", statistic[0][0], " - ", statistic[0][1])
        print("-"*80)
        
        print("COSINE SIMILARITY")
        print("\tFrom the Concatenate Last Four Method: %.3f" % statistic[1][0], 
              "\n\tFrom the Sum Last Four Method: %.3f" % statistic[1][1],
              "\n\tFrom the Get Second Last Method: %.3f" % statistic[1][2])
        print("\nDOT PRODUCT")
        print("\tFrom the Concatenate Last Four Method: %.3f" % statistic[2][0], 
              "\n\tFrom the Sum Last Four Method: %.3f" % statistic[2][1],
              "\n\tFrom the Get Second Last Method: %.3f" % statistic[2][2])
        print()

def process_single_word(word, word_vector_array):
    
    all_stats = []

    # Initialize a variable to hold the vector
    word_vectors_of_target_word = []
    
    # Populate a new array of word vectors for the target word
    for entry in word_vector_array:
        if entry[1] == word:
            word_vectors_of_target_word.append(entry[3])

    # Perform statistics on each permutation
    #! Note, the original sentence is lost.
    i = 0
    while i < len(word_vectors_of_target_word):
        j = i+1
        while j < len(word_vectors_of_target_word):
            cosine = cosine_similarity(word_vectors_of_target_word[i], word_vectors_of_target_word[j])
            dot = dot_product(word_vectors_of_target_word[i], word_vectors_of_target_word[j])
            statistic = [[word, word], cosine, dot]   # [word, word] so it is compatible with the print_statistics()
            all_stats.append(statistic)

            j += 1
        i += 1

    return all_stats

def print_to_csv_experiment_2(experiment_results):
    
    file = open("Experiment2_%s.csv" % checkpoint_name, 'w', newline='')
    with file:

        # Format and add the header
        tempDictKeys = {"Word Pair":0,
                        "Cosine: Cat Last Four":0,
                        "Cosine: Sum Last Four":0,
                        "Cosine: Get 2nd Last":0,
                        "Dot: Cat Last Four":0,
                        "Dot: Sum Last Four":0,
                        "Dot: Get 2nd Last":0}
        write = csv.writer(file, tempDictKeys.keys())
        write.writerow(tempDictKeys)

        # Create a temporary dictionary entry to print the 3D array
        tempDict = {}
        for statistic in experiment_results:
            tempDict = {"Word Pair":statistic[0],
                        "Cosine: Cat Last Four":statistic[1][0],
                        "Cosine: Sum Last Four":statistic[1][1],
                        "Cosine: Get 2nd Last":statistic[1][2],
                        "Dot: Cat Last Four":statistic[2][0],
                        "Dot: Sum Last Four":statistic[2][1],
                        "Dot: Get 2nd Last":statistic[2][2]}
            
            write.writerow(tempDict.values())

def print_to_csv_experiment_3(experiment_results):

    file = open("Experiment3_%s_%s.csv" % (experiment_results[0][0][0], checkpoint_name), 'w', newline='')
    with file:

        # Format and add the header
        tempDictKeys = {"Contexts Compared":0,
                        "Cosine: Cat Last Four":0,
                        "Cosine: Sum Last Four":0,
                        "Cosine: Get 2nd Last":0,
                        "Dot: Cat Last Four":0,
                        "Dot: Sum Last Four":0,
                        "Dot: Get 2nd Last":0}
        write = csv.writer(file, tempDictKeys.keys())
        write.writerow(tempDictKeys)

        # Create custom labels
        labels = ["Definition - Reflection1",
                  "Definition - Reflection2",
                  "Definition - Non-Context",
                  "Definition - Dark Humor",
                  "Reflection1 - Reflection2",
                  "Reflection1 - Non-Context",
                  "Reflection1 - Dark Humor",
                  "Reflection2 - Non-Context",
                  "Reflection2 - Dark Humor",
                  "Non-Context - Dark Humor"]
        iterate = 0

        # Create a temporary dictionary entry to print the 3D array
        tempDict = {}
        for statistic in experiment_results:
            tempDict = {"Contexts Compared":labels[iterate],
                        "Cosine: Cat Last Four":statistic[1][0],
                        "Cosine: Sum Last Four":statistic[1][1],
                        "Cosine: Get 2nd Last":statistic[1][2],
                        "Dot: Cat Last Four":statistic[2][0],
                        "Dot: Sum Last Four":statistic[2][1],
                        "Dot: Get 2nd Last":statistic[2][2]}
            
            write.writerow(tempDict.values())
            iterate += 1

def main():
    
    # Data strutures to store word embeddings
    word_vector_array_ex1 = []
    word_vector_array_ex2 = []

    # List to store the output for Experiment 2
    experiment2_output = []

    # Access the text file to construct the input text and target words
    with open('input.txt') as file:
        
        line = file.readline()

        if line == "experiment2_input\n":
            
            line = file.readline()
            while line != "\n":
                
                # Clean up in the input text
                experiment = line
                experiment = experiment.split('|')
                input_text = experiment[0]
                target_word = experiment[1].strip()

                input = [input_text, target_word]

                # Return word vectors
                word_vectors = get_word_vectors(input[0], input[1])
                word_vector_array_ex2.append(word_vectors)

                # Process the next input line
                line = file.readline()
              
            # Skip the empty formatting line
            line = file.readline()
        
        if line == "word_pairs\n":

            line = file.readline()
            while line != "\n":
                
                target_words = [word.strip() for word in line.split(',')]

                print(target_words)

                # Calculate statistics for the target words
                statistic = process_word_pair(target_words, word_vector_array_ex2)
                experiment2_output.append(statistic)

                # Process the next word pair
                line = file.readline()
            
            # Print statistics for the experiment
            print_statistics(experiment2_output)
            print_to_csv_experiment_2(experiment2_output)

            # Skip the empty formatting line
            line = file.readline()
            
        else:
            print("Error: no word-pairs words provided.")
        
        if line == "experiment3_input\n":
            
            line = file.readline()
            while line != "\n":
                
                # Clean up in the input text
                experiment = line
                experiment = experiment.split('|')
                input_text = experiment[0]
                target_word = experiment[1].strip()

                input = [input_text, target_word]

                # Return word vectors
                word_vectors = get_word_vectors(input[0], input[1])
                word_vector_array_ex1.append(word_vectors)

                # Process the next input line
                line = file.readline()
              
            # Skip the empty formatting line
            line = file.readline()
      
        else:
            print("Error: experiment 1 was not considered.")
        
        if line == "single_words\n":

            line = file.readline()
            while line:

                target_word = line.strip()

                # Calculate statistics for all uses of this word in input sentences
                single_word_statistics = process_single_word(target_word, word_vector_array_ex1)

                # Print the statistics
                print_single_word_statistics(single_word_statistics)
                print_to_csv_experiment_3(single_word_statistics)

                # Process the next single word
                line = file.readline()

        else:
            print("Error: no multi-context words provided.")
        
        # Meta stats
        print("No. of decompose errors: %d" % error_count)

if __name__ == "__main__":
    main()


