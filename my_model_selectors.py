import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        # call the super class init function
        super().__init__(all_word_sequences,all_word_Xlengths,this_word,n_constant,min_n_components,max_n_components,random_state,verbose)

        # save a list of the candidate models as a dictonary keyed by BIC score
        self.model_cands = dict()

        # for each value of 'n' states to try, train a model for the input word and calculate the BIC score
        # and add model to our candidates dictionary list

        for n in range(self.min_n_components,self.max_n_components):
            try:
                model = self.base_model(n)

                # N is the number of data points.  get that by summing up the lengths of the X vectors
                N=math.fsum(self.lengths)

                # p = num params = # transprobs + # means + # covars
                # calculate using the size of the appropriate array shapes.  There are easier, less 'generic' ways
                # to do this of course.   Just try to be generic to the returned model

                p = model.transmat_.shape[0] * model.transmat_.shape[1] + \
                    model.means_.shape[0] * model.means_.shape[1] + \
                    model.covars_.shape[0] * model.covars_.shape[1]

                # L = the Log Likelihood score of this model
                L =model.score(self.X,self.lengths)

                # calculate the BIC score.   This uses a negative Log Likelihood, so we'll need minimize it later
                thisscore= (-2.0 * L) + (p * math.log(N))
                self.model_cands[thisscore] = model
            except:
                pass

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # if we didn't get any valid model candidates, return a default model
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        #otherwise find the minimum BIC score calculated from the candidate dictionary keys
        minscore=min(self.model_cands.keys())
        # and return that model as the best
        return self.model_cands[minscore]
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):

        # call the super class init function
        super().__init__(all_word_sequences,all_word_Xlengths,this_word,n_constant,min_n_components,max_n_components,random_state,verbose)

        # save a list of the candidate models as a dictonary keyed by BIC score
        self.model_cands = dict()

        # for each value of 'n' states to try, train a model for the input word and calculate the DIC score
        # and add model to our candidates dictionary list

        for n in range(self.min_n_components,self.max_n_components):
            try:
                # train a model for the input word and get the Log Likelihood score

                model = self.base_model(n)
                myscore =model.score(self.X,self.lengths)

                # initialize variables used to calculate an average log likelihood of all the other words to get an
                # 'anti-likelihood' score

                othertotalscore = 0.0
                otherwords = self.words.keys()
                othermodelcount = len(otherwords) - 1
                # for each other word (except the current word we're training on), get a model for it and get
                # its log likelihood, and add into our running total

                for w in otherwords:
                    if w == this_word:
                        continue
                    try:
                        ms = ModelSelector(self.words, self.hwords, w, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose )
                        othermodel = ms.base_model(n)
                        try:
                            otherscore = othermodel.score(ms.X, ms.lengths)
                            othertotalscore += otherscore
                        except:
                            pass
                    except:
                        pass
                # calculate an average log likelihood score of all the other words and subtract it from this word log likelihood
                # score for a final DIC score.   use the DIC value as the model candidate dictionary key

                avgotherscore = othertotalscore/othermodelcount
                thisscore = myscore - avgotherscore
                self.model_cands[thisscore] = model
            except:
                pass

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        # if we didn't get any valid model candidates, return a default model
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        #otherwise find the maximum DIC score calculated from the candidate dictionary keys
        maxscore=max(self.model_cands.keys())
        # and return that model
        return self.model_cands[maxscore]
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):

        # call the super class init function
        super().__init__(all_word_sequences,all_word_Xlengths,this_word,n_constant,min_n_components,max_n_components,random_state,verbose)

        # save a list of the candidate models as a dictonary keyed by BIC score
        self.model_cands = dict()

        # for each value of 'n' states to try, train a model for the input word and calculate the and
        # averge the log likelihood scores for the returned training set returned for each fold split
        # then add model to our candidates dictionary list

        for n in range(self.min_n_components,self.max_n_components):
            # initialize some variables used to calculate the average log likelihood scores
            scoretotal=0.0
            avgscore = 0.0
            numscores=0
            # find a minimum number of folds to use.  either use 3 or the number of sequences we have.
            folds=min(len(self.sequences),3)
            # if we have less than 2 folds we can split into, just train a single model and use its log likelihood.
            if folds < 2:
                try:
                    model = self.base_model(n)
                    avgscore=model.score(self.X,self.lengths)
                except:
                    pass
            else:
                # otherwise, create the kfold object with an appropriate number of folds not to break the split function
                split_method = KFold(folds)

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    word_sequences, word_Xlengths = combine_sequences(cv_train_idx,self.sequences)

                    try:
                        # then for each fold split, combine the entries assigned to the training set
                        # and use that to score the trained model against, than accumlate the returned
                        # log likelihoods to calculate an overall average score for all the folds
                        model = self.base_model(n)
                        thisscore=model.score(word_sequences,word_Xlengths)
                        scoretotal += thisscore
                        numscores += 1
                    except:
                        pass
                        model = None

                # if we had at least one valid model returned, calculate the average score.
                if numscores > 0:
                    avgscore = scoretotal/numscores
            # if we got at least one model,save it as a candidate keyed by the average log likelihood of the fold scores
            if model is not None:
                self.model_cands[avgscore] = model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # if we didn't get any valid model candidates, return a default model
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        #otherwise find the maximum CV average log likelihood score calculated from the candidate dictionary keys
        maxscore=max(self.model_cands.keys())
        # and return that model
        return self.model_cands[maxscore]
        raise NotImplementedError
