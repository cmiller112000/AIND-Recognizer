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
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorBIC,self).__init__(all_word_sequences,all_word_Xlengths,this_word,3,min_n_components,max_n_components,random_state,verbose)
        self.model_cands = dict()
        for n in range(self.min_n_components,self.max_n_components):
            try:
                model = self.base_model(n)
                N=math.fsum(self.lengths)
                p = (n * (n-1)) + (n * (model.n_features-1))
                L =model.score(self.X,self.lengths)
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
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        minscore=min(self.model_cands.keys())
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
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorDIC,self).__init__(all_word_sequences,all_word_Xlengths,this_word,3,min_n_components,max_n_components,random_state,verbose)
        self.model_cands = dict()
        for n in range(self.min_n_components,self.max_n_components):
            try:
                model = self.base_model(n)
                myscore =model.score(self.X,self.lengths)
                othertotalscore = 0.0
                othermodelcount = 0.0
                otherwords = self.words.keys()
                othermodelcount = len(otherwords) - 1
                for w in otherwords:
                    if w == this_word:
                        continue
                    try:
                        ms = ModelSelector(self.words, self.hwords, w, 3, self.min_n_components, self.max_n_components, self.random_state, self.verbose )
                        othermodel = ms.base_model(n)
                        try:
                            otherscore = othermodel.score(ms.X, ms.lengths)
                            othertotalscore += otherscore
                        except:
                            pass
                    except:
                        pass
                avgotherscore = othertotalscore/othermodelcount
                thisscore = myscore - avgotherscore
                self.model_cands[thisscore] = model
            except:
                pass

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        maxscore=max(self.model_cands.keys())
        return self.model_cands[maxscore]
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        super(SelectorCV,self).__init__(all_word_sequences,all_word_Xlengths,this_word,3,min_n_components,max_n_components,random_state,verbose)
        self.model_cands = dict()
        for n in range(self.min_n_components,self.max_n_components):
            scoretotal=0.0
            avgscore = 0.0
            numscores=0
            folds=min(len(self.sequences),3)
            split_method = KFold(folds)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                word_sequences, word_Xlengths = combine_sequences(cv_train_idx,self.sequences)

                try:
                    model = self.base_model(n)
                    thisscore=model.score(word_sequences,word_Xlengths)
                    scoretotal += thisscore
                    numscores += 1
                except:
                    pass
                    model = None
            if numscores > 0:
                avgscore = scoretotal/numscores
            if model is not None:
                self.model_cands[avgscore] = model

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        if len(self.model_cands) == 0:
            return self.base_model(self.min_n_components)
        maxscore=max(self.model_cands.keys())
        return self.model_cands[maxscore]
        raise NotImplementedError
