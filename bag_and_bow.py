import numpy as np
alphabet = {
    'a' : 1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 
    'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14,
    'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21,
    'v':22, 'w':23, 'x':24, 'y':25, 'z':26
}

def bag(word: str):   
    word = word.lower()
    bag = []
    
    for i in range(255):
        if i < len(word):
            try:
                bag.append(alphabet[word[i]]/26)
            except:
                bag.append(0)
        else:
            bag.append(0)
    return bag


def bow(label:str, labels: list[str], show_details=False) -> list[int]:
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(labels)
    for i, l in enumerate(labels):
        if l == label:
            # assign 1 if current word is in the vocabulary position
            bag[i] = 1
            if show_details:
                print("found: '%s' in: %s" % (label, labels))
                print("at index %i" % (i))
    return bag
