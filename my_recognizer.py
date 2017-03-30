import warnings

from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # get all the XItems for the input test set
    XItems = test_set.get_all_Xlengths()
    for item in XItems:
        # then for each Xitem, get the X and lengths
        X,lengths = XItems[item]
        # initialize some working variables
        probentries=dict()
        maxLogL = None
        bestguess=""
        for w in models:
            # for each word in our list of trained models, determine the log likelihood for running the current
            # X sequences and lengths against this words model to see if we have potential match

            try:
                logL, state_seq = models[w].decode(X,lengths,algorithm='viterbi')
                probentries[w] = logL
                # keep track of the model with the highest log likelihood as our best guess
                if maxLogL is None or logL > maxLogL:
                    maxLogL = logL
                    bestguess=w
            except:
                probentries[w] = None
                pass
        # accumulate our best match log likelihood and the best guess word for it.
        probabilities.append(probentries)
        guesses.append(bestguess)

    return probabilities, guesses
    raise NotImplementedError
