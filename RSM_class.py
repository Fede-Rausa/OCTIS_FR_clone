import numpy as np
from tqdm import tqdm

class RSM(object):

    def __init__(self):
        self.W = None

    def softmax(self, array):
        exparr = np.exp(array)
        return exparr/exparr.sum()

    def softmax0(self, array):
        exparr = np.exp(array)
        return exparr/np.tile(exparr.sum(axis=1), (exparr.shape[1],1)).T

    def sigmoid(self,x):
        return (1/(1+np.exp(-x)))

############## energy and probability

    def neg_energy(self, v, h):
        w_vh, w_v, w_h = self.W
        D = v.sum()
        en = v @ w_v + D * h @ w_h + v @ w_vh @ h
        return en
        
    def neg_energy0(self, v, h):
        w_vh, w_v, w_h = self.W
        D = v.sum(axis=1)
        t1 = v @ w_v
        t2 = D * (h @ w_h)
        t3 = (v @ w_vh @ h.T).sum(axis=1)
        en = t1 + t2 + t3
        return en
    
    def neg_free_energy(self, v):  #it's equivalent to the log pdf
        w_vh, w_v, w_h = self.W
        T = self.hidden
        D = v.sum()
        fren = np.dot(v, w_v)
        for j in range(T):
            w_j = w_vh[:,j]
            a_j = w_h[j]
            fren += np.log(1 + np.exp(D*a_j + np.dot(v,w_j)))
        return fren
    
    def neg_free_energy0(self, v):  #it's equivalent to the log pdf
        w_vh, w_v, w_h = self.W
        T = self.hidden
        D = v.sum(axis=1)
        fren = np.dot(v, w_v)
        for j in range(T):
            w_j = w_vh[:,j]
            a_j = w_h[j]
            fren += np.log(1 + np.exp(D*a_j + np.dot(v,w_j)))
        return fren

    def marginal_pdf(self, v):
        return np.exp(self.neg_free_energy(v))

    def marginal_pdf0(self, v):
        return np.exp(self.neg_free_energy0(v))

    def visible2hidden(self, v):
        w_vh, w_v, w_h = self.W
        D = v.sum()
        energy = D*w_h + np.dot(v, w_vh)
        return self.sigmoid(energy)
        #probs = self.sigmoid(energy)
        #return probs
    
    def visible2hidden0(self,v):
        w_vh, w_v, w_h = self.W
        D = np.tile(v.sum(axis=1), (w_h.shape[0], 1)).T
        energy = D*w_h + np.dot(v, w_vh)
        return self.sigmoid(energy)
        #D = v.sum(axis=1)
        #D = np.tile(D, (w_h.shape[0], 1)).T
        #probs = self.sigmoid(energy)
        #return probs


    def hidden2visible(self, h): 
        #in this function I haven't to scale up for the number of words
        w_vh, w_v, w_h = self.W
        energy = w_v+ np.dot(w_vh, h)
        return self.softmax(energy)
        #probs = self.softmax(energy.T)
        #return probs

    def hidden2visible0(self, h):
        w_vh, w_v, w_h = self.W
        #print(h.shape[0])
        energy = np.tile(w_v, (h.shape[0], 1)).T + np.dot(w_vh, h.T)
        return self.softmax0(energy.T)
        # energy = np.outer(w_v, np.ones(h.shape[0])) + np.dot(w_vh, h.T)
        # probs = self.softmax(energy.T)
        # return probs

    def topic_words(self, topk, id2word=None):
        w_vh, w_v, w_h = self.W
        T = self.hidden
        if id2word==None:
            id2word = self.id2word
        words = np.array([k for k in id2word.token2id.keys()])

        toplist = []
        for t in range(T):
            topw = w_vh[: , t]
            bestwords = words[np.argsort(topw)[::-1]][0:topk]
            toplist.append(bestwords)

        return toplist


################# sampling


    def multinomial_sample(self, probs, N):
        if (np.any(np.isnan(probs))):
            print(probs)
        return np.random.multinomial(N, probs, size=1)[0]

    def unif_reject_sample(self,probs):
        h_unif = np.random.rand(*probs.shape)
        h_sample = np.array(h_unif < probs, dtype=int)
        return h_sample

    def deterministic_sample(self,probs):
        return (probs > 0.5).astype(int)
    
    def gibbs_transition(self, v):
        D = v.sum()
        hidden_probs = self.visible2hidden(v)
        hidden_sample = self.unif_reject_sample(hidden_probs)
        visible_probs = self.hidden2visible(hidden_sample)
        visible_sample = self.multinomial_sample(visible_probs, D)
        return visible_sample


    def gibbs_transition0(self, v):
        D = v.sum(axis=1)
        hidden_probs = self.visible2hidden0(v)
        hidden_sample = self.unif_reject_sample(hidden_probs)
        visible_probs = self.hidden2visible0(hidden_sample)
        visible_sample = np.empty(v.shape)
        for i in range(v.shape[0]):
            visible_sample[i] = self.multinomial_sample(visible_probs[i], D[i])
        return visible_sample


    def MH_transition(self, state, logpdf):
        new = self.gibbs_transition(state)
        old_logpdf = logpdf(state)
        new_logpdf = logpdf(new)

        accept_ratio = min(1, np.exp(new_logpdf - old_logpdf))

        # Accept or reject
        if np.random.random() < accept_ratio:
            return new
        else:
            return state


    def MH_transition0(self, state, logpdf):
        new = self.gibbs_transition0(state)
        old_logpdf = logpdf(state)
        new_logpdf = logpdf(new)

        accept_ratio = min(1, np.exp(new_logpdf - old_logpdf))

        # Accept or reject
        if np.random.random() < accept_ratio:
            return new
        else:
            return state


################# training


    
    def gradient_step(self, v1, v2, h1, h2):
        w_vh, w_v, w_h = self.W
        vel_vh, vel_v, vel_h = self.velocities
        m = self.momentum
        lr = self.lr
        vel_vh = vel_vh * m + np.dot(v1.T, h1) - np.dot(v2.T, h2)
        vel_v = vel_v * m + v1.sum(axis=0) - v2.sum(axis=0)
        vel_h = vel_h * m + h1.sum(axis=0) - h2.sum(axis=0)
        w_vh += vel_vh * lr
        w_v += vel_v * lr
        w_h += vel_h * lr
        self.W = w_vh, w_v, w_h
        self.velocities = vel_vh, vel_v, vel_h

    def cd_step0(self, v, K, mean_h = True):
        v0 = v
        h0 = self.visible2hidden0(v0)

        v1 = v0
        for k in range(K):
            v1 = self.gibbs_transition0(v1)
        h1 = self.visible2hidden0(v1)

        if not mean_h:  #converting probabilities to binaries
            h0 = self.unif_reject_sample(h0)
            h1 = self.unif_reject_sample(h1)

        self.gradient_step(v0,v1,h0,h1)


    def cd_step(self, v0, K, mean_h=True):
        ndocs = v0.shape[0]
        h0 = np.empty((v0.shape[0], self.hidden))
        v1 = v0 #initialize transition
        h1 = h0
        for d in range(ndocs):
            h0[d] = self.visible2hidden(v0[d])
            for k in range(K):
                v1[d] = self.gibbs_transition(v1[d])
            h1[d] = self.visible2hidden(v1[d])

        if not mean_h:  #converting probabilities to binaries
            h0 = self.unif_reject_sample(h0)
            h1 = self.unif_reject_sample(h1)

        self.gradient_step(v0,v1,h0,h1)     




    def train(self, dtm, hidden=5, epochs=3, btsz=100, 
              lr=0.01, momentum=0.5, K=1, 
              epochs_per_monitor=1, monitor = False,
              softstart=0.001, initw=None, val_dtm=None):

        self.momentum = momentum
        self.lr = lr
        self.hidden = hidden
        self.visible = dtm.shape[1]
        
        doval = (val_dtm is not None)

        ##init

        N, dictsize = dtm.shape
        batches = int(np.floor(N/btsz))

        if initw is not None:
            self.W = initw

        if self.W is None:
            w_vh = softstart * np.random.randn(dictsize, hidden)
            w_v = softstart * np.random.randn(dictsize)
            w_h = softstart * np.random.randn(hidden)
        else:
            print('train already available weights')
            w_vh, w_v, w_h = self.W

        vel_vh = np.zeros((dictsize, hidden))
        vel_v = np.zeros((dictsize))
        vel_h = np.zeros((hidden))

        self.W = w_vh, w_v, w_h
        self.velocities = vel_vh, vel_v, vel_h

        monit_epochs = np.arange(stop = epochs, step = epochs_per_monitor)
        next_monitor = 0
        self.train_loglik = np.empty(len(monit_epochs))
        self.train_ppl = np.empty(len(monit_epochs))
        if doval:
            self.val_loglik = np.empty(len(monit_epochs))
            self.val_ppl = np.empty(len(monit_epochs))

        ##loop
        for t in tqdm(range(epochs)):
            start_id = 0
            for b in range(batches):
                v = dtm[start_id : start_id + btsz , :]
                self.cd_step0(v, K)
                start_id += btsz

            if monitor:
                if t == monit_epochs[next_monitor]:
                    next_monitor += 1
                    next_monitor = t + epochs_per_monitor

                    self.train_loglik[t] = np.mean(self.neg_free_energy0(dtm))
                    self.train_ppl[t] = self.log_ppl_upo(dtm)

                    if doval:
                        self.val_loglik[t] = np.mean(self.neg_free_energy0(val_dtm))
                        self.val_ppl[t] = self.log_ppl_upo(val_dtm)




############ perplexity and probability


    def log_ppl_upbo(self, dtm):
        """
        return the log perplepxity upper bound 
        given a document term matrix
        """
        mfh = self.visible2hidden0(dtm)
        vprob = self.hidden2visible0(mfh)
        lpub = np.exp(-np.nansum(np.log(vprob)*dtm)/np.sum(dtm))
        return lpub



    def approx_ppl(self, testmatrix):

        w_vh, w_v, w_h = self.W
        D=testmatrix.sum(axis=1)

        # compute hidden activations
        h = self.sigmoid(np.dot(testmatrix, w_vh) + np.outer(D, w_h))

        # compute visible activations
        v = np.dot(h, w_vh.T) + w_v
        pdf = self.softmax0(v)

        #compute the per word perplexity
        z = np.nansum(testmatrix * np.log(pdf))
        s = np.sum(D)
        ppl = np.exp(- z / s)
        return ppl



    def approx_prob(self, dtm):
        w_vh, w_v, w_h = self.W
        D = dtm.sum(axis=1)
        # compute hidden activations
        h = self.sigmoid(np.dot(dtm, w_vh) + np.outer(D, w_h))

        # compute visible activations
        v = np.dot(h, w_vh.T) + w_v
        pdf = self.softmax0(v)

        return pdf


    def ais(self, S=1000, niter=100, D=20, MH_steps=0):
        T = self.hidden
        Za = 2**T
        K = self.visible #voacb length
        #inverse temperature values
        beta = np.arange(start=0, stop=1+1/S, step=1/S)

        #intermediate pdf
        def temp_pdf(docvec, b):
            return np.exp(b*np.log(self.marginal_pdf(docvec)))
        
        def log_temp_pdf(docvec, b):
            return b * self.neg_free_energy(docvec)
            #return b*np.log(self.marginal_pdf(docvec))

        #w_ais_list = np.empty(niter)

        log_w_ais_list = np.empty(niter)
        for it in tqdm(range(niter)):

            v_sampled = np.random.multinomial(D, np.ones(K)/K, size=1)[0]

            #loop
            log_w_ais = 0  #w_ais = 1
            for s in range(S-1):
                

                if (MH_steps>0):
                    def lpd(doc): return log_temp_pdf(doc, beta[s])
                    for m in range(MH_steps):
                        v_sampled = self.MH_transition(v_sampled, logpdf=lpd)
                else:
                    v_sampled = self.gibbs_transition(v_sampled)
                
                logratio = log_temp_pdf(v_sampled, beta[s+1]) - log_temp_pdf(v_sampled, beta[s])
                if not np.isnan(logratio):
                    log_w_ais = log_w_ais + logratio
                #ratio = temp_pdf(v_sampled, beta[s+1])/temp_pdf(v_sampled, beta[s])
                #w_ais = w_ais*ratio

            log_w_ais_list[it] = log_w_ais


        vec = log_w_ais_list - np.log(log_w_ais_list.shape[0])
        log_avg_ratio = np.max(vec) + np.log(np.sum(np.exp(vec - np.max(vec))))
        
        var_log_ratio = np.nanvar(log_w_ais_list)

        log_Zb = log_avg_ratio + np.log(Za)
        #Zb = np.exp(log_Zb)

        return log_Zb, Za, log_avg_ratio, var_log_ratio


