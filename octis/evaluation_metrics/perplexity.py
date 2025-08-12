import numpy as np
from octis.evaluation_metrics.metrics import AbstractMetric
from gensim import corpora

class Perplexity(AbstractMetric):
    def __init__(self, dataset, id2word=None, use_validation = False, return_crossentropy=False):  # use_train=False, normalize=False
        """
        
        dataset : octis.dataset.dataset.Dataset
            The dataset instance, to get test corpus.

        return_crossentropy: bool = False
            if the log of perplexity (crossentropy) should be returned instead of the ppl
        use_validation : bool = True
            argument passed to get_partitioned_corpus

        (args not implemented)
        normalize : bool = False
            if the probability matrices in model_output should be normalized
        use_train : bool = False
            if the score should be computed on the train set

        """
        super().__init__()
        self.dataset = dataset
        self.return_crossentropy = return_crossentropy
        self.test_corpus = self.dataset.get_partitioned_corpus(use_validation = use_validation)[1]   # test documents
        self.id_corpus = self.corpus_to_bow(self.test_corpus, id2word)

    def corpus_to_bow(self, corpus, id2word=None):
        if (id2word==None):
            id2word = corpora.Dictionary(corpus)
        bow = [id2word.doc2bow(document) for document in corpus]
        return bow


    def score(self, model_output):
        """
        Parameters
        ----------
        model_output : dict
            Must contain 'topic-word-matrix' and 'test-topic-document-matrix'.

        Returns
        -------
        float : Perplexity score if return_crossentropy=False, otherwise crossentropy (the log of perplexity)
        details:

        Computes the perplexity of a DIRECTED topic model given the output and dataset.
        Assumes the model_output contains 'topic-word-matrix' and 'test-topic-document-matrix'.

        Given D a corpus, that is a list of M documents
        Given Q a probability model that returns P(topic_k|doc)  and P(word|topic_k)
        
        the perplexity formula:
        ppl(D, Q) = exp(-sum_docs(sum_words_in_doc(P(word|doc))) / sum_docs(sum_words_in_doc(1)) ) =
        = exp(-sum_d(sum_i(P(word_i|doc_d))) / sum_d(N_d)

        were d is the index of the docs, i is the index of the i word of the doc, i=1,..,N_d, 
        N_d is the number of words in the doc, and were

        P(word|doc) = sum_k(P(topic_k|doc) * P(word|topic_k))

        You can show that perplexity is the geometric mean of P(word|doc) for all the words inside D
        ppl(D,Q) = prod_wd( P(word_w|doc_d) ) ^ (-1/sum_d(N_d))

        the crossentropy formula:
        ce = H(P, Q) = log(ppl(D,Q)) = H(P) + KL(P||Q)
        were D ~ P, P is the real distribution of the data, H(P) = H(P,P) is the entropy of the data,
        KL(P||Q) is the Kullback-Leibler div between the real dist P and the estimated model Q

        """
        # Get test topic-document matrix and topic-word matrix
        td_mat = model_output["test-topic-document-matrix"]  # shape: (num_topics, num_test_docs)
        tw_mat = model_output["topic-word-matrix"]             # shape: (num_topics, vocab_size)

        bow = self.id_corpus
        loglik = 0
        nw = 0
        for d in range(len(bow)):
            doc = bow[d]
            for w, count in doc:
                prob_t_given_d = td_mat[:,d]
                prob_w_given_t = tw_mat[:,w]
                prob_w_given_d = prob_t_given_d @ prob_w_given_t
                loglik += np.log(prob_w_given_d) * count
                nw += count

        logppl = - np.sum(loglik) / np.sum(nw)

        if self.return_crossentropy:
            return logppl
        else:
            ppl = np.exp(logppl)
            return ppl

        
        

