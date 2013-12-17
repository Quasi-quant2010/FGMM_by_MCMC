# -*- coding: utf-8 -*-

import matplotlib.mlab
import numpy as np
import gzip
import cPickle
import codecs
import operator
import re
import sys
import os
import cPickle
import time
import numpy.lib.recfunctions
import itertools
import matplotlib.pyplot as plt
import math

def gmm(x, p, mean, precision):
    def normal_dist(x, mu, sigma):
        tmp = ( 1.0/(math.sqrt(2.0*np.pi)*sigma) ) * \
            np.exp( -pow((x-mu),2) / (2.0*pow(sigma,2))  )
        return tmp
    #p*normal_dist(x,mu1,sigma1) + (1-p)*normal_dist(x,mu2,sigma2)
    return sum( [ p[j]*normal_dist(x,mean[j],precision[j]) for j in xrange(len(p))] )

def gaussian(y, mu, precision):
    # precision = 1 / sigma^2
    return math.sqrt(precision) * np.exp( -0.5 * precision * pow((y-mu),2) )


def GMM_Collapse_Gibbs_Sampling(iterations, N_int, K_int, 
                                mps, 
                                precision, 
                                alpha,
                                mean_0, precision_0,
                                mu_lower, mu_higher):
    """
    [argmemts]
    iterations : mcmcを実行する回数
    N_int : the number of data
    K_int : the number of cluster
    mps : the number of cluster paramters
    precision : gmm 1/sigma^2
    alpha : hyperprameters for mean
    mean_0, precision_0
    mu_lower : 
    mu_higher :
    
    [Index]
    K_int : the number of classes
    N_int : the number of datas

    [Parameters]
    z_iteration_n : the class label k for the data x_i (iteration=1,...,iteratons, n=1,...,N)
    p_z_i_k : the probability that z_i takes part in the class k (iterationに対し、リストを作成)
    pi_iteration_k : The Weight For the class k over the class K, p(z_i=k|pai), (iteratons, k=1,...,K)
    theta_iteration_k : the parameter vector for the class k, (iteration=1,...,iteratons, k=1,...,K)
    alpha : the hyperparameters for p, in this process fixed
    beta : the hyper parameters for theta, in this process fixed

    [storage]
    z : matrix(iterations*N) : the latent classes
    x : vector(1*N) : the observed datas(N)
    p_z_i_k : vector(1:N) : the probability that z_i takes part in the class k
    n : vector(1*K) : p(x|pai) = \PI_{k=1}^{K} pai_k^{n_k}
    pi : matrix(1*K) : the mixing weight for each cluster k
    theta : matrix(iterations*mps) : the cluster parameters for each iterations
    lh ; vector(1*iterations) : likelihood, \sum_{k=1}^{K}
    """
    # make storage
    z = [ np.array([0 for i in xrange(N_int)])
          for iteration in xrange(iterations) ]
    n = [ 0  for k in xrange(K_int) ]
    pi = np.array([0.0  for k in xrange(K_int)])
    theta = [ np.array([0.0  for mp in xrange(mps)])
              for iteration in xrange(iterations) ]
    lh = [0.0 for iteration in xrange(iterations)]

    # initialize
    # data[j,:] : j行目
    # data[:,k] : k列目
    np.random.seed(1)    
    theta[0] = np.random.uniform( mu_lower, mu_higher, K_int )
    z[0] = np.random.randint(K_int, size=N_int)

    # collapse gibbs sampling
    for iteration in xrange(1,iterations):
        tmp_lh =  0.0
        # Step1
        for i in xrange(N_int):
            # Step1.1 : z(iteration-1)からi番目の要素を除く
            tmp_z = list(z[iteration-1])[:i] + list(z[iteration-1])[i+1:]
            tmp_z = np.array(tmp_z)

            # Step1.2 : generate p( z_i^{t}=j | p_j^{t-1}, theta_j^{t-1}, x_i)
            p_z_i_k = [0.0 for k in xrange(K_int)]
            for k in xrange(K_int):
                n_k_minus_i = len( np.where(tmp_z==k)[0] )
                weight = (float(n_k_minus_i)+(alpha/K_int)) / (float(N_int)+alpha-1.0)
                p_z_i_k[k] = weight * \
                    gaussian(y=x[i], mu=theta[iteration-1][k], precision=precision)
                del weight, n_k_minus_i
            del tmp_z
        
            # Step1.3 :  z_i^{iteration}を更新
            p_z_i_k = np.array(p_z_i_k)            
            z[iteration][i] = p_z_i_k.argmax()# indexがcluster numberと一致
            
            # Step1.4 : likelihood(z_i|z_{-i},・)を計算
            tmp_lh += -np.log(p_z_i_k.sum())
            del p_z_i_k

        # 尤度を更新
        lh[iteration] = tmp_lh
        del tmp_lh

        # Step2 : クラスターパラメータ更新        
        for k in xrange(K_int):
            index = np.where(z[iteration]==k)
            mean_x_k = x[index].mean()
            if np.isfinite(mean_x_k)==False:
                # indexが存在しない場合の処理
                mean_x_k= 0.0
            mean = ( mean_x_k*len(index[0])*precision + mean_0*precision_0) / \
                (len(index[0])*precision + precision_0)
            sigma = 1.0 / \
                (len(index[0])*precision + precision_0)
            theta[iteration][k] = np.random.normal(mean,sigma)

    # iteration = iterationsのおいて生成されたz_{i=1}^{N}を用いて、n・piを計算
    # Step0.1 : 求めたしたz[iteration]を用いてnを計算
    for k in xrange(K_int):
        index = np.where(z[iterations-1]==k)[0]
        #nを更新
        n[k] = len(index)
        del index    

    # Step0.2 : nからpiを生成
    """
    np.random.dirichletに極端なパラメータを与えると
    まれにNaNやInfを返すことがある為、再度サンプリング
    """
    pi = np.random.dirichlet(
        tuple([float(n[k])+alpha/K_int
               for k in xrange(K_int)])
        )
    tmp_boolean = (np.isfinite(pi)).astype(int)
    while reduce(lambda x,y:x*y, tmp_boolean)==0:
        pi = np.random.dirichlet(
            tuple([float(n[k])+alpha/K_int
                   for k in xrange(K_int)])
            )
        tmp_boolean = (np.isfinite(pi)).astype(int)
    del tmp_boolean

    del z, n    
    
    return {'lh':lh, 'pi':pi, 'theta':theta}

def artificial_data(x, mean, precision, n):
    """
    mean = [100,20,50]
    precision = [1/15, 1/5, 1/30]
    n = [300, 200, 500]
    """    
    return np.hstack( np.random.normal(*args) for args in zip(mean,precision,n) )

if __name__ == "__main__":    

    for K_int in xrange(2,4):

        print 'Cluster Number is:', K_int

        np.random.seed(10)
        N_int = 1000 #observation number
        #K_int = 2 #cluster number
        p = np.random.uniform(0,1,K_int)
        p = p /p.sum()

        # inference index
        iterations = 100
        mps = K_int # model_paramters_size, 今ガウス分布の平均値のみ推定の対象としているため
        alpha = 0.3 # hyperparameters, the summetric Dirichlet distribution <- ここを可変にする
        mean_0 = 2.0 # hyperparameters
        precision_0 = 1.0 # hyperparameters

        # 人工データ生成
        """
        mu_lower = np.random.randint(-100,100)
        if mu_lower<0:
            mu_higher = -mu_lower
        else:
            mu_higher = mu_lower + np.random.randint(0,100)
        """
        mean = np.random.randint(-30,30,K_int)
        precision = [1.0 for k in xrange(K_int)]
        n = np.random.multinomial(N_int, p)
        x = artificial_data(x = np.random.uniform(-35,
                                                  35,
                                                  N_int),
                            mean = mean,
                            precision = precision,
                            n = n)
        # True Parameters
        true_result = {'true_theta':mean, 'true_pi':p}
        
        plt.hist(x,
                 bins = np.linspace(-35.0, 35.0, num=100),
                 normed = True)    
        plt.hold(True)    
        plt.plot(np.linspace(-35.0, 35.0, num=100),
                 gmm(x = np.linspace(-35.0, 35.0, num=100),
                     p = p,
                     mean = mean,
                     precision = precision),
                 color='red', linewidth=2)
        plt.title(','.join(map(str,n)))
        plt.savefig('collapse_gibbs_true_cluster_' + str(K_int) + '.png')    
        plt.close()        
        del mean, precision, n

        # パラメータ推定
        """
        def GMM_Collapse_Gibbs_Sampling(iterations, N_int, K_int,
                                        mps,
                                        alpha, precision,
                                        mean_0, precision_0)
        """
        result = GMM_Collapse_Gibbs_Sampling(iterations=iterations,
                                             N_int=N_int,
                                             K_int=K_int,
                                             mps=mps,
                                             alpha=alpha, precision=1,
                                             mean_0=mean_0, precision_0=precision_0,
                                             mu_lower=-30, mu_higher=30)
        savefile = 'Likelihood_MixingRate_ClusterParameters_cluster_' + str(K_int) + '.dump.gz'        
        
        for key in true_result.keys():
            result[key] = true_result[key]

        print result['true_theta'], result['theta']

        with codecs.getreader("utf-8")( gzip.open(savefile, mode='w') ) as outfile:
            cPickle.dump(result, outfile, 2)        
        del savefile        
        
        #-----  graph ------
        
        # 1. likehood
        plt.plot(
            map(int, 
                np.linspace(1, len(result['lh']), num=len(result['lh'])-1)), result['lh'][1:],
            'b.-',
            ms=10
            )
        plt.title(r'$\sum_{i=1}^{N} ln \sum_{k=1}^{K} p(z_i=j|z_{-i}=j,x,\theta,\alpha,\lambda)$')
        plt.xlabel('Iteration')
        plt.ylabel('Likelihood')
        fpng = 'collapse_gibbs_likelihood_cluster_' + str(K_int) + '.png'
        plt.savefig(fpng)
        plt.close()
    

        del result, theta, true_result
        
