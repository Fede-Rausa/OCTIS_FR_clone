import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns



## funzioni utili

def TrainTestSplit(data, train_prop, return_ids = False, return_mask=False, print_shapes=False):
    """
    data: numpy ndarray with the datapoints on the axis 0
    train_prop: float in (0,1), ratio of the data that will go in the training
    return_ids: boolean, if the interger indexes of training should be returned
    return_mask: boolean, if the boolean mask of the training should be returned
    
    returns tuple:
    - train array
    - test array
    - train_id if return_ids=True
    - train_mask if return_mask=True
    """
    N = data.shape[0]
    rows_id = np.arange(N)
    train_id = np.random.choice(rows_id, size=int(N*train_prop), replace=False)
    train_mask = np.isin(rows_id, train_id)
    train = data[train_mask]
    test = data[~train_mask]
    output = (train, test)
    if (print_shapes):
        print('train shape: '+str(train.shape))
        print('test shape: '+str(test.shape))

    if (return_ids):
        output + (train_id,)
    if (return_mask):
        output + (train_mask,)
    return output


def unlist(LIST):
    return [item for sublist in LIST for item in sublist]

def softmax_vec(array):
    exparr = np.exp(array)
    probs = exparr/exparr.sum()
    return probs

def strong_softmax_vec(array):
    maxs = np.max(array)
    lse = maxs + np.log(np.sum(np.exp(array - maxs)))
    probs = np.exp(array - lse)
    return probs

def softmax_mat(array):
    exparr = np.exp(array)
    return exparr/exparr.sum(axis=1)

def strong_softmax_mat(array):
    maxs = np.max(array, axis=1, keepdims=True)
    lse = maxs + np.log(np.sum(np.exp(array - maxs), axis=1, keepdims=True))
    return np.exp(array - lse)

def sigmoid(x):
    return (1/(1+np.exp(-x)))


def summary(vec):
    return {
        'dim' : vec.shape,
        'mean' : np.round(np.mean(vec), 2),
        'stdev' : np.round(np.sqrt(np.var(vec)), 2),
        'min' : np.round(np.min(vec), 2),
        'q25%' : np.round(np.quantile(vec, 0.25),2),
        'median': np.round(np.median(vec), 2),
        'q75%' : np.round(np.quantile(vec, 0.75),2),
        'max' : np.round(np.max(vec), 2)
    }



############ plot numpy arrays

import pandas as pd
import seaborn as sns


def hist(vec, xlab='x'):
    df = pd.DataFrame({xlab:vec})
    pl = sns.histplot(data=df, x=xlab)
    return pl

def scatter(x, y, xlab='x', ylab='y'):
    df = pd.DataFrame({xlab:x, ylab:y})
    pl = sns.scatterplot(data=df, x=xlab, y=ylab)
    return pl

def regscatter(x, y, xlab='x', ylab='y', 
               order=1, lowess=False, 
               ci=None, color=None):
    df = pd.DataFrame({xlab:x, ylab:y})
    pl = sns.regplot(data=df, x=xlab, y=ylab, scatter=True, 
                     fit_reg=True, color = color,
                     order=order, lowess=lowess, ci=ci)
    return pl


def plotline(y,x=None, xlab='x', ylab='y', title=''):
    """
    y: value of the series, represented by a line
    x: index order of value 
    example usage
    plotline(y=np.random.rand(45), x=np.random.rand(45), ylab='rand1', xlab='rand2')
    """

    if (x is None):
        x = np.arange(len(y))
    df = pd.DataFrame({xlab:x, ylab:y})
    pl = sns.lineplot(data=df, y=ylab, x=xlab)
    pl.set_title(title)
    return  pl


def multilineplot(df, colnames=None, xlab='index', ylab='series', title=''):
    if colnames is not None:
        df = df[colnames]
    df[xlab] = np.arange(df.shape[0])
    df0 = pd.melt(df, [xlab])
    pl = sns.lineplot(data=df0, x=xlab, y='value', hue='variable', 
                        markers=True, style='variable')
    pl.set_title(title)
    pl.set_xlabel(xlab)
    pl.set_ylabel(ylab)
    return  pl




##################### NLP functions

import gensim.corpora as corpora
from tqdm import tqdm


def get_vocab(tokenized_corpus):
    
    id2word = corpora.Dictionary(tokenized_corpus)
    return id2word


def build_dtm(tokenized_corpus, id2word = None):
    """
    converts a tokenized corpus to a DOcument Term Matrix. id2word is a gensim dictionary.
    """
    if (id2word == None):
        id2word = corpora.Dictionary(tokenized_corpus)
    else:
        id2word = id2word
    id_corpus = [id2word.doc2bow(document) for document in tokenized_corpus]
    vocab = id2word.token2id
    N = len(id_corpus)
    DTM = np.zeros((N, len(vocab)))
    for i in tqdm(range(N)):
        doc = id_corpus[i]
        for id, count in doc:
            DTM[i,id] = count

    return DTM


def dtm_to_bow(dtm):
    """
    Convert a DTM to BoW format
    """
    bows = []
    for row in dtm:
        bow = [(i, int(count)) for i, count in enumerate(row) if count > 0]
        bows.append(bow)
    return bows


def corpus_to_bow(corpus, id2word=None):
    """
    Convert a tokenized corpus to BoW format
    id2word is a gensim dictionary
    """
    if (id2word==None):
        id2word = corpora.Dictionary(corpus)
    bow = [id2word.doc2bow(document) for document in corpus]
    return bow






######### linear regression models

class ols(object):

    def train(self, x, y, intercept=True, w=None):
        self.intercept = intercept
        if intercept:
            # Add intercept column if not present
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        
        self.onevar = (len(x.shape)==1)

        n = y.shape[0]

        if self.onevar:
            p = 1
            self.coefs = np.sum(y)/np.sum(x)
            self.fitted = x * self.coefs
            self.gram = 1 / np.dot(x,x)
        else:
            p = x.shape[1]
            Q, R = np.linalg.qr(x)
            Rinv = np.linalg.inv(R)
            self.coefs = Rinv @ Q.T @ y
            self.fitted = x @ self.coefs
            self.H = Q @ Q.T
            self.gram = Rinv @ Rinv.T

        self.df = n - p
        self.residuals = y - self.fitted
        self.mse = np.mean(self.residuals**2)
        self.mae = np.mean(np.abs(self.residuals))
        self.TSS = np.sum((y-np.mean(y))**2)
        self.RSS = np.sum(self.residuals**2) #(self.residuals @ self.residuals)
        self.R2 = 1 - self.RSS/self.TSS
        self.variance =  self.RSS / self.df
        self.s = np.sqrt(self.variance)

        if self.onevar:
            self.stderrors = np.sqrt(self.gram*self.variance)
        else:
            self.stderrors =  np.sqrt(np.diag(self.gram) * self.variance)
            
        self.tstats = self.coefs / self.stderrors
        self.pvalues = 1 - sc.stats.t.cdf(np.abs(self.tstats), df=self.df)



    def predict(self, x):
        if self.intercept:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        if self.onevar:
            return x * self.coefs
        else:
            return x @ self.coefs


    def summary(self):

        self.perf = {
            'R2' : self.R2,
            'mse': self.mse,
            'mae': self.mae,
            's' : self.s}

        if not self.onevar:
            self.coeftable = pd.DataFrame({
                'coef': self.coefs,
                'stderror' : self.stderrors,
                'tstat' : self.tstats,
                'pv': self.pvalues
            })
        else:
            self.coeftable = {
                'coef': self.coefs,
                'stderror' : self.stderrors,
                'tstat' : self.tstats,
                'pv': self.pvalues
            }
    

        return self.perf, self.coeftable



class wls(object):

    def train(self, x, y, intercept=True, w=None):
        self.intercept = intercept
        if intercept:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        
        self.onevar = (len(x.shape) == 1)
        n = y.shape[0]

        # --- Weighted transformation ---
        if w is not None:
            w = np.asarray(w)
            if w.ndim != 1 or w.shape[0] != n:
                raise ValueError("Weights w must be a 1D array of length n")
            W_sqrt = np.sqrt(w)
            x = x * W_sqrt[:, None]
            y = y * W_sqrt

        if self.onevar:
            p = 1
            self.coefs = np.sum(y) / np.sum(x)
            self.fitted = x * self.coefs
            self.gram = 1 / np.dot(x, x)
        else:
            p = x.shape[1]
            Q, R = np.linalg.qr(x)
            Rinv = np.linalg.inv(R)
            self.coefs = Rinv @ Q.T @ y
            self.fitted = (np.hstack((np.ones((n, 1)), x[:, 1:])) 
                           if intercept else x) @ self.coefs
            self.H = Q @ Q.T
            self.gram = Rinv @ Rinv.T

        self.df = n - p
        self.residuals = y / (np.sqrt(w) if w is not None else 1) - self.fitted  # back to original scale if weighted
        self.mse = np.mean(self.residuals ** 2)
        self.mae = np.mean(np.abs(self.residuals))
        self.TSS = np.sum((y / (np.sqrt(w) if w is not None else 1) - np.mean(y / (np.sqrt(w) if w is not None else 1))) ** 2)
        self.RSS = np.sum(self.residuals ** 2)
        self.R2 = 1 - self.RSS / self.TSS
        self.variance = self.RSS / self.df
        self.s = np.sqrt(self.variance)

        if self.onevar:
            self.stderrors = np.sqrt(self.gram * self.variance)
        else:
            self.stderrors = np.sqrt(np.diag(self.gram) * self.variance)
            
        self.tstats = self.coefs / self.stderrors
        self.pvalues = 1 - sc.stats.t.cdf(np.abs(self.tstats), df=self.df)


    def predict(self, x):
        if self.intercept:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            x = np.hstack((np.ones((x.shape[0], 1)), x))

        if self.onevar:
            return x * self.coefs
        else:
            return x @ self.coefs


    def summary(self):

        self.perf = {
            'R2' : self.R2,
            'mse': self.mse,
            'mae': self.mae,
            's' : self.s}

        if not self.onevar:
            self.coeftable = pd.DataFrame({
                'coef': self.coefs,
                'stderror' : self.stderrors,
                'tstat' : self.tstats,
                'pv': self.pvalues
            })
        else:
            self.coeftable = {
                'coef': self.coefs,
                'stderror' : self.stderrors,
                'tstat' : self.tstats,
                'pv': self.pvalues
            }
    

        return self.perf, self.coeftable

