import sys

from model import WordPairClassifier

if __name__ == "__main__":
    modelfile = sys.argv[1]

    print("Loading model...")
    model = WordPairClassifier.load(modelfile)
    word2id = model.config["word2id"]

    while True:
        print("Enter two words to calculate hyponymy:")
        word_pair = input().strip()
        assert(len(word_pair.split()) == 2), "Need to input exactly two words"
        word1 = word_pair.split()[0]
        word2 = word_pair.split()[1]

        if word1 in word2id and word2 in word2id:
            cost, scores = model.test([word2id[word1]], [word2id[word2]], [[]], [0])        
            print(word1 + "\t" + word2 + "\t" + str(scores[0]))
        else:
            print(word1 + "\t" + word2 + "\tNot in vocabulary")
