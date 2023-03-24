# Import libraries
import torch
import torchtext
import gensim.downloader
import numpy
import csv

# Import the pre-trained models
fasttext = torchtext.vocab.FastText(language="en")
glove = torchtext.vocab.GloVe(name="42B")
    # 6B: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors)
    # 42B: Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors)
    # 840B: Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors)
    # 2B: Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors)
word2vec = gensim.downloader.load('word2vec-google-news-300')
    # 100B: Google News (3 million words and phrases, 300d vectors)


def cosine_similarity(target_words):

    # Glove and Fasttext uses the torchtext library; word2vec requires numpy 
    cosine_glove = torch.cosine_similarity(glove[target_words[0]].unsqueeze(0), glove[target_words[1]].unsqueeze(0))
    cosine_fasttext = torch.cosine_similarity(fasttext[target_words[0]].unsqueeze(0), fasttext[target_words[1]].unsqueeze(0))
    cosine_word2vec = numpy.dot(word2vec[target_words[0]], word2vec[target_words[1]])/(numpy.linalg.norm(word2vec[target_words[0]])* numpy.linalg.norm(word2vec[target_words[1]]))

    return [float(cosine_glove), float(cosine_fasttext), float(cosine_word2vec)]

def dot_product(target_words):
    
    # Glove and Fasttext uses the torchtext library; word2vec requires numpy 
    dot_glove = torch.dot(glove[target_words[0]], glove[target_words[1]])
    dot_fasttext = torch.dot(fasttext[target_words[0]], fasttext[target_words[1]])
    dot_word2vec = numpy.dot(word2vec[target_words[0]], word2vec[target_words[1]])

    return [float(dot_glove), float(dot_fasttext), float(dot_word2vec)]

def print_statistics(experiment_outputs):

    # Print a header
    print("PRINTING STATISTICS FOR WORD PAIRS")
    print("-"*80, '\n')

    for statistic in experiment_outputs:
        
        # Identify the word pair 
        print("WORD PAIR: ", statistic[0][0], " - ", statistic[0][1])
        print("-"*80)

        # Print the statistics
        print("COSINE SIMILARITY")
        print("\tFrom GloVe: %.4f" % statistic[1][0], 
              "\n\tFrom Fasttext: %.4f" % statistic[1][1],
              "\n\tFrom word2vec: %.4f" % statistic[1][2])
        print("\nDOT PRODUCT")
        print("\tFrom GloVe: %.4f" % statistic[2][0], 
              "\n\tFrom Fasttext: %.4f" % statistic[2][1],
              "\n\tFrom word2vec: %.4f" % statistic[2][2])
        print()

def main():

    # Store the statistics for each word pair. 
    # This will be passed when printing the statistics.
    experiment_outputs = []

    # Access the text file to construct the input text and target words
    with open('input.txt') as file:
       
        # Read the first line
        line = file.readline()
        
        # Skip the parts of the text file that are irrelevant to experiment2
        while line != "word_pairs\n":
            line = file.readline()
        
        # Skip the line, "word_pairs\n"
        line = file.readline()

        while line != "\n":
            
            # Get the word pair
            target_words = [word.strip() for word in line.split(',')]
            
            # Calculate the cosine similarity and dot product for all models
            cosine = cosine_similarity(target_words)
            dot = dot_product(target_words)

            # Save the results and include the word-pair for later reference
            statistic = [target_words, cosine, dot]
            experiment_outputs.append(statistic)

            line = file.readline()
        
        # Print the result of the experiment to the terminal
        print_statistics(experiment_outputs)

        # Print the results to a .csv file
        file = open('Experiment1.csv', 'w', newline='')
        with file:
            
            # Format and add the header
            tempDictKeys = {"Word Pair":0, 
                      "Cosine Similarity: GloVe":0, 
                      "Cosine Similarity: Fasttext":0,
                      "Cosine Similarity: word2vec":0,
                      "Dot Product: GloVe":0,
                      "Dot Product: Fasttext":0,
                      "Dot Product: word2vec":0}
            write = csv.writer(file, tempDictKeys.keys())
            write.writerow(tempDictKeys)

            # Create a temporary dictionary entry to print the 3D array
            tempDict = {}
            for statistic in experiment_outputs:
                tempDict = {"Word Pair": statistic[0], 
                            "Cosine Similarity: GloVe": statistic[1][0],
                            "Cosine Similarity: Fasttext": statistic[1][1],
                            "Cosine Similarity: word2vec": statistic[1][2],
                            "Dot Product: GloVe": statistic[2][0],
                            "Dot Product: Fasttext": statistic[2][1],
                            "Dot Product: word2vec": statistic[2][2]}
                
                write.writerow(tempDict.values())

if __name__ == "__main__":
    main()

# Links
'''
torchtext documentation | https://pytorch.org/text/stable/index.html
Glove | https://nlp.stanford.edu/projects/glove/
Fasttext | https://pytorch.org/text/stable/vocab.html#fasttext 
word2vec | https://github.com/RaRe-Technologies/gensim-data
'''