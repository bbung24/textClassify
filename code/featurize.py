#!/usr/bin/env python

# This program takes the given text input and computes the tf-idf rank

import sys
import math
import numpy
from collections import defaultdict
def featurize(train_articles, test_articles):

    def make_features(articles):
        STOPWORDS = ['a','able','about','across','after','all','almost','also',
                    'am','among','an','and','any','are','as','at','be',
                    'because','been','but','by','can','cannot','could','dear',
                    'did','do','does','either','else','ever','every','for',
                    'from','get','got','had','has','have','he','her','hers',
                    'him','his','how','however','i','if','in','into','is','it',
                    'its','just','least','let','like','likely','may','me',
                    'might','most','must','my','neither','no','nor','not','of',
                    'off','often','on','only','or','other','our','own','rather',
                    'said','say','says','she','should','since','so','some',
                    'than','that','the','their','them','then','there','these',
                    'they','this','to','too','us','wants','was','we','were',
                    'what','when','where','which','while','who','whom','why',
                    'will','with','would','yet','you','your',"'s", "'ve"]

        line_count = 0
        unique = set()

        # Generate all the unique words in the entire document
        for line in articles:
            line_count += 1
            unique.update([word for word in line.split() 
                        if not word in STOPWORDS])

        articles.seek(0)

        df = [0] * len(unique)
        idf = [0] * len(unique)

        # Generate the text and document frequency per article
        all_tf = []
        lines = 0
        for line in articles:           
            lines += 1
            if lines%100 == 0:
                print "Line " + str(lines)
            i = 0
            tf_temp = defaultdict(int)
            for word in line.split():
                tf_temp[word] += 1

            for word in unique:                
                if(tf_temp[word] > 0):
                    df[i] += 1
                i += 1
            all_tf.append([tf_temp[word] for word in unique])
            

        # Generate the inverse document frequency
        i = 0
        idf = [0] * len(unique)
        for term in df:
            idf[i] = math.log(line_count/term)
            i += 1

        a = numpy.array(all_tf)
        b = numpy.array(idf)

        # # Multiplies tf and idf to create tfidf
        tfidf = a*b
        articles.close()
        return tfidf

    return (make_features(train_articles), make_features(test_articles))

def articles_to_features(train_in_name, train_out_name, 
                        test_in_name, test_out_name):
    train_in_file = open(train_in_name, 'r')
    test_in_file = open(test_in_name, 'r')

    train_features, test_features = featurize(train_in_file, test_in_file)

    for features, out_name in ((train_features, train_out_name),
                               (test_features, test_out_name)):
        with open(out_name, 'wb') as csvfile:
            # Output features in MATLAB-readable CSV format
            for row in features:
                csvfile.write(', '.join([str(feature) for feature in row]))
                csvfile.write('\n')

    csvfile.close()

if __name__ == '__main__':
    articles_to_features(*sys.argv[1:5])
