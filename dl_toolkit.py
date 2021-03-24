import numpy as np
import math
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import roc_curve,auc
from scipy import interp
import copy
class MLPClassifier(object):
    def __init__(self,**kwargs):
        self.__layers=kwargs['layers']
        if 'learning_rate' in kwargs:
            self.__learning_rate=kwargs['learning_rate']
        else:
            self.__learning_rate=1e-5
        if 'activation_function' in kwargs:
            self.__activation_function=kwargs['activation_function']
        else:
            self.__activation_function='relu'
        if 'optimizer' in kwargs:
            self.__optimizer=kwargs['optimizer']
        else:
            self.__optimizer='gradient_descent'
        if 'Weight_init' in kwargs:
            self.__Weight_init=kwargs['Weight_init']
        else:
            self.__Weight_init='random'
        self.__initialize_weights()
        if self.__optimizer=='momentum':
            self.__gamma=0.9
            self.__vw=[np.zeros_like(CC) for CC in self.__weights]
            self.__vb=[np.zeros_like(CC) for CC in self.__bias]
        elif self.__optimizer=='adagrad':
            self.__epsilon=sys.float_info.epsilon
            self.__vw=[np.zeros_like(CC) for CC in self.__weights]
            self.__vb=[np.zeros_like(CC) for CC in self.__bias]
        elif self.__optimizer=='rmsprop':
            self.__epsilon=sys.float_info.epsilon
            self.__gamma=0.9
            self.__vw=[np.zeros_like(CC) for CC in self.__weights]
            self.__vb=[np.zeros_like(CC) for CC in self.__bias]
        elif self.__optimizer=='adam':
            self.__epsilon=sys.float_info.epsilon
            self.__beta=0.9
            self.__gamma=0.999
            self.__mw=[np.zeros_like(CC) for CC in self.__weights]
            self.__vw=[np.zeros_like(CC) for CC in self.__weights]
            self.__mb=[np.zeros_like(CC) for CC in self.__bias]
            self.__vb=[np.zeros_like(CC) for CC in self.__bias]
            self.__t=1
        else:
            self.__vw=[np.zeros_like(CC) for CC in self.__weights]
            self.__vb=[np.zeros_like(CC) for CC in self.__bias]
            self.__gamma=0.9
        if 'gamma' in kwargs:
            self.__gamma=kwargs['gamma']
        if 'beta' in kwargs:
            self.__beta=kwargs['beta']
        if 'convergence_threshold' in kwargs:
            self.__convergence_threshold=kwargs['convergence_threshold']
        else:
            self.__convergence_threshold=sys.float_info.epsilon
        if 'Regularization' in kwargs:
            self.__Regularization=kwargs['Regularization']
        else:
            self.__Regularization='l2'
        if 'lamda' in kwargs:
            self.__lamda=kwargs['lamda']
        else:
            self.__lamda=0.1
        if 'Batch_size' in kwargs:
            self.__Batch_size=kwargs['Batch_size']
        else:
            self.__Batch_size=64
        if 'Num_epochs' in kwargs:
            self.__Num_epochs=kwargs['Num_epochs']
        else:
            self.__Num_epochs=100
        if 'dropout' in kwargs:
            self.__dropout=kwargs['dropout']
            self.__masks=[]
        else:
            self.__dropout=0
    def __initialize_weights(self):
        print('\nInitializing weights...\n')
        self.__weights=[[] for i in range(len(self.__layers)-1)]
        self.__bias=[[] for i in range(len(self.__layers)-1)]
        if self.__Weight_init=='random':
            for i in range(len(self.__layers)-1):
                self.__weights[i]=np.random.rand(self.__layers[i],self.__layers[i+1])-0.5
                self.__bias[i]=np.random.rand(1,self.__layers[i+1])-0.5
        elif self.__Weight_init=='xavier':
            for i in range(len(self.__layers)-1):
                self.__weights[i]=(np.random.randn(self.__layers[i],self.__layers[i+1])-0.5)*np.sqrt(1/(self.__layers[i]+self.__layers[i+1]))
                self.__bias[i]=(np.random.randn(1,self.__layers[i+1])-0.5)*np.sqrt(1/(self.__layers[i]+self.__layers[i+1]))
        else:
            for i in range(len(self.__layers)-1):
                self.__weights[i]=(np.random.randn(self.__layers[i],self.__layers[i+1])-0.5)*np.sqrt(2/(self.__layers[i]+self.__layers[i+1]))
                self.__bias[i]=(np.random.randn(1,self.__layers[i+1])-0.5)*np.sqrt(2/(self.__layers[i]+self.__layers[i+1]))
    def __activate(self,instance,function):
        if function=='relu':
            return np.maximum(instance,0)
        elif function=='sigmoid':
            return 1/(1+np.exp(-instance))
        elif function=='softmax':
            deno=np.sum(np.exp(instance))
            return np.exp(instance)/deno
        else:
            return np.tanh(instance)
    def __activation_derivative(self,instance,function):
        if function=='sigmoid':
            s=self.__activate(instance,function)
            return s*(1-s)
        elif function=='relu':
            instance[instance<=0]=0
            instance[instance>0]=1
            return instance
        else:
            t=self.__activate(instance,function)
            return 1-t*t
    def __drop_activations(self,A):
        drop=np.random.binomial(1,1-self.__dropout,size=A.shape)
        self.__masks.append(drop)
        if self.__dropout!=0:
            A/=1-self.__dropout
        A*=drop
        return A
    def __forward_pass(self,features):
        Z=[[] for i in range(len(self.__layers)-1)]
        A=[[] for i in range(len(self.__layers)-1)]
        intermediate=np.matmul(features,self.__weights[0])+self.__bias[0]
        Z[0]=intermediate
        if len(self.__layers)!=2:
            for j in range(intermediate.shape[0]):
                intermediate[j]=self.__activate(intermediate[j],self.__activation_function)
            A[0]=intermediate
        for i in range(1,len(self.__layers)-1):
            intermediate=np.matmul(intermediate,self.__weights[i])+self.__bias[i]
            Z[i]=intermediate
            if i!=len(self.__layers)-2:
                for j in range(intermediate.shape[0]):
                    intermediate[j]=self.__activate(intermediate[j],self.__activation_function)
                A[i]=intermediate
        for i in range(intermediate.shape[0]):
            intermediate[i]=self.__activate(intermediate[i],'softmax')
        A[len(self.__layers)-2]=intermediate
        return intermediate,Z,A
    def __dropout_forward_pass(self,features):
        Z=[[] for i in range(len(self.__layers)-1)]
        A=[[] for i in range(len(self.__layers)-1)]
        intermediate=np.matmul(features,self.__weights[0])+self.__bias[0]
        Z[0]=intermediate
        if len(self.__layers)!=2:
            for j in range(intermediate.shape[0]):
                intermediate[j]=self.__activate(intermediate[j],self.__activation_function)
            intermediate=self.__drop_activations(intermediate)
            A[0]=intermediate
        for i in range(1,len(self.__layers)-1):
            intermediate=np.matmul(intermediate,self.__weights[i])+self.__bias[i]
            Z[i]=intermediate
            if i!=len(self.__layers)-2:
                for j in range(intermediate.shape[0]):
                    intermediate[j]=self.__activate(intermediate[j],self.__activation_function)
                intermediate=self.__drop_activations(intermediate)
                A[i]=intermediate
        for i in range(intermediate.shape[0]):
            intermediate[i]=self.__activate(intermediate[i],'softmax')
        A[len(self.__layers)-2]=intermediate
        return intermediate,Z,A
    def __update_weights(self,dW,dB):
        if self.__optimizer=='gradient_descent':
            for i in range(len(self.__layers)-1):
                self.__weights[i]-=self.__learning_rate*dW[i]
                self.__bias[i]-=self.__learning_rate*dB[i]
        elif self.__optimizer=='momentum':
            for i in range(len(self.__layers)-1):
                self.__vw[i]=self.__gamma*self.__vw[i]+self.__learning_rate*dW[i]
                self.__vb[i]=self.__gamma*self.__vb[i]+self.__learning_rate*dB[i]
                self.__weights[i]-=self.__vw[i]
                self.__bias[i]-=self.__vb[i]
        elif self.__optimizer=='adagrad':
            for i in range(len(self.__layers)-1):
                self.__vw[i]+=dW[i]*dW[i]
                self.__vb[i]+=dB[i]*dB[i]
                self.__weights[i]-=self.__learning_rate/(np.sqrt(self.__vw[i])+self.__epsilon)*dW[i]
                self.__bias[i]-=self.__learning_rate/(np.sqrt(self.__vb[i])+self.__epsilon)*dB[i]
        elif self.__optimizer=='rmsprop':
            for i in range(len(self.__layers)-1):
                self.__vw[i]=self.__gamma*self.__vw[i]+(1-self.__gamma)*dW[i]*dW[i]
                self.__vb[i]=self.__gamma*self.__vb[i]+(1-self.__gamma)*dB[i]*dB[i]
                self.__weights[i]-=self.__learning_rate/(np.sqrt(self.__vw[i])+self.__epsilon)*dW[i]
                self.__bias[i]-=self.__learning_rate/(np.sqrt(self.__vb[i])+self.__epsilon)*dB[i]
        elif self.__optimizer=='adam':
            for i in range(len(self.__layers)-1):
                self.__mw[i]=self.__beta*self.__mw[i]+(1-self.__beta)*dW[i]
                self.__vw[i]=self.__gamma*self.__vw[i]+(1-self.__gamma)*dW[i]*dW[i]
                self.__mb[i]=self.__beta*self.__mb[i]+(1-self.__beta)*dB[i]
                self.__vb[i]=self.__gamma*self.__vb[i]+(1-self.__gamma)*dB[i]*dB[i]
                mw_cap=self.__mw[i]/(1-np.power(self.__beta,self.__t))
                vw_cap=self.__vw[i]/(1-np.power(self.__gamma,self.__t))
                mb_cap=self.__mb[i]/(1-np.power(self.__beta,self.__t))
                vb_cap=self.__vb[i]/(1-np.power(self.__gamma,self.__t))
                self.__weights[i]-=self.__learning_rate/(np.sqrt(vw_cap)+self.__epsilon)*mw_cap
                self.__bias[i]-=self.__learning_rate/(np.sqrt(vb_cap)+self.__epsilon)*mb_cap
                self.__t+=1
        else:
            for i in range(len(self.__layers)-1):
                self.__vw[i]=self.__gamma*self.__vw[i]+self.__learning_rate*dW[i]
                self.__vb[i]=self.__gamma*self.__vb[i]+self.__learning_rate*dB[i]
                self.__weights[i]-=self.__learning_rate*dW[i]
                self.__bias[i]-=self.__learning_rate*dB[i]
    def __dropout_backward_pass(self,X,Z,A,ground_truth):
        dZ=[[] for i in range(len(self.__layers)-1)]
        dW=[[] for i in range(len(self.__layers)-1)]
        dB=[[] for i in range(len(self.__layers)-1)]
        T=np.zeros((ground_truth.shape[0],self.__layers[len(self.__layers)-1]))
        for i in range(ground_truth.shape[0]):
            T[i][ground_truth[i]]=1;
        m=Z[0].shape[0]
        for i in range(len(self.__layers)-2,-1,-1):
            if i==len(self.__layers)-2:
                dZ[i]=A[i]-T
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(A[i-1].T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
            elif i==0:
                dA=dZ[i+1].dot(self.__weights[i+1].T)
                dA/=1-self.__dropout
                dA*=self.__masks[i]
                dZ[i]=dA*self.__activation_derivative(Z[i],self.__activation_function)
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((X.T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((X.T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(X.T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
            else:
                dA=dZ[i+1].dot(self.__weights[i+1].T)
                dA/=1-self.__dropout
                dA*=self.__masks[i]
                dZ[i]=dA*self.__activation_derivative(Z[i],self.__activation_function)
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(A[i-1].T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
        return dW,dB
    def __backward_pass(self,X,Z,A,ground_truth):
        dZ=[[] for i in range(len(self.__layers)-1)]
        dW=[[] for i in range(len(self.__layers)-1)]
        dB=[[] for i in range(len(self.__layers)-1)]
        T=np.zeros((ground_truth.shape[0],self.__layers[len(self.__layers)-1]))
        for i in range(ground_truth.shape[0]):
            T[i][ground_truth[i]]=1;
        m=Z[0].shape[0]
        for i in range(len(self.__layers)-2,-1,-1):
            if i==len(self.__layers)-2:
                dZ[i]=A[i]-T
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(A[i-1].T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
            elif i==0:
                dZ[i]=dZ[i+1].dot(self.__weights[i+1].T)*self.__activation_derivative(Z[i],self.__activation_function)
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((X.T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((X.T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(X.T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
            else:
                dZ[i]=dZ[i+1].dot(self.__weights[i+1].T)*self.__activation_derivative(Z[i],self.__activation_function)
                if self.__Regularization=='l2':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*self.__weights[i])
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*self.__bias[i])
                elif self.__Regularization=='l1':
                    dW[i]=(1/m)*((A[i-1].T).dot(dZ[i])+self.__lamda*np.sign(self.__weights[i]))
                    dB[i]=(1/m)*(np.sum(dZ[i],axis=0)+self.__lamda*np.sign(self.__bias[i]))
                else:
                    dW[i]=(1/m)*(A[i-1].T).dot(dZ[i])
                    dB[i]=(1/m)*np.sum(dZ[i],axis=0)
        return dW,dB
    def __form_batches(self,X,Y):
        batches=math.ceil(Y.shape[0]/self.__Batch_size)
        feature_batches=np.array_split(X,batches,axis=0)
        target_batches=np.array_split(Y,batches,axis=0)
        return batches,feature_batches,target_batches
    def __loss(self,ground_truth,predicted):
        q=np.array([])
        for i in range(predicted.shape[0]):
            g=np.zeros(predicted.shape[1])
            g[ground_truth[i]]=1
            q=np.append(q,[np.sum(np.multiply(g,np.log(predicted[i]))*(-1))])
        q=np.mean(q)
        if self.__Regularization=='l2':
            l2=0
            for i in range(len(self.__layers)-1):
                l2+=np.sum(np.square(self.__weights[i]))
            return q+self.__lamda/2/len(ground_truth)*l2
        elif self.__Regularization=='l1':
            l1=0
            for i in range(len(self.__layers)-1):
                l1+=np.sum(np.abs(self.__weights[i]))
            return q+self.__lamda/len(ground_truth)*l1
        return q
    def fit(self,X,Y,Xv,Yv):
        print('Training starts...\n')
        X=X/np.max(X)
        Xv=Xv/np.max(Xv)
        previous_loss=0
        batches,feature_batches,target_batches=self.__form_batches(X,Y)
        self.__training_loss=[]
        self.__epoch_list=[]
        self.__validation_loss=[]
        best_score=0
        best_weights=[]
        best_bias=[]
        for i in range(self.__Num_epochs):
            for j in range(batches):
                if self.__optimizer=='nag':
                    for k in range(len(self.__layers)-1):
                        self.__weights[k]-=self.__gamma*self.__vw[k]
                        self.__bias[k]-=self.__gamma*self.__vb[k]
                if self.__dropout==0:
                    predicted,Z,A=self.__forward_pass(feature_batches[j])
                    dW,dB=self.__backward_pass(feature_batches[j],Z,A,target_batches[j])
                else:
                    predicted,Z,A=self.__dropout_forward_pass(feature_batches[j])
                    dW,dB=self.__dropout_backward_pass(feature_batches[j],Z,A,target_batches[j])
                self.__masks=[]
                self.__update_weights(dW,dB)
            if self.__optimizer=='adam':
                self.__t=1
            loss=self.__loss(Y,self.__forward_pass(X)[0])
            score=self.score(Xv,Yv)*100
            print('Epochs {}/{} ==============>> Training Accuracy: {:.2f}%,  Validation Accuracy: {:.2f}%'.format(i+1,self.__Num_epochs,self.score(X,Y)*100,score))
            if score>best_score:
                best_score=score
                best_weights=copy.deepcopy(self.__weights)
                best_bias=copy.deepcopy(self.__bias)
            self.__training_loss.append(loss)
            self.__epoch_list.append(i+1)
            self.__validation_loss.append(np.mean(self.__loss(Yv,self.__forward_pass(Xv)[0])))
            if abs(loss-previous_loss)<self.__convergence_threshold:
                print('\nNeural Net has converged...\n')
                break
            previous_loss=loss
        self.__weights=copy.deepcopy(best_weights)
        self.__bias=copy.deepcopy(best_bias)
    def predict(self,X):
        X=X/np.max(X)
        predicted,_,_=self.__forward_pass(X)
        return np.argmax(predicted,axis=1)
    def predict_proba(self,X):
        X=X/np.max(X)
        predicted,_,_=self.__forward_pass(X)
        return predicted
    def get_params(self):
        return self.__weights,self.__bias
    def score(self,X,y):
        X=X/np.max(X)
        predicted,_,_=self.__forward_pass(X)
        c=0
        for i in range(predicted.shape[0]):
            if np.argmax(predicted[i])==y[i]:
                c+=1
        return c/predicted.shape[0]
    def loss_plot(self):
        plt.plot(self.__epoch_list,self.__training_loss,'-o',label='Training Cross Entropy Loss')
        plt.plot(self.__epoch_list,self.__validation_loss,'-o',label='Validation Cross Entropy Loss')
        plt.legend(loc='upper right')
        plt.gcf().set_size_inches((15,15))
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Vs. Epoch')
        plt.show()
    def confusion_matrix_plot(self,ground_truth,predicted):
        plt.figure(figsize=(15,15))
        sn.heatmap(confusion_matrix(ground_truth,predicted),annot=True,fmt='d',cmap='YlGnBu',linewidths=0.5)
        plt.show()
    def roc_plot(self,X,Y):
        X=X/np.max(X)
        classes=list(set(Y.tolist()))
        n_classes=len(classes)
        Y_original=list()
        for i in range(len(Y)):
            g=np.zeros(n_classes)
            g[Y[i]]=1
            Y_original.append(g)
        Y_original=np.array(Y_original)
        predicted,_,_=self.__forward_pass(X)
        fpr,tpr,roc_auc=dict(),dict(),dict()
        for i in range(n_classes):
            fpr[i],tpr[i],_=roc_curve(Y_original[:,i],predicted[:,i])
            roc_auc[i]=auc(fpr[i],tpr[i])
        all_fpr=np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr=np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr+=interp(all_fpr,fpr[i],tpr[i])
        plt.figure(figsize=(15,15))
        for i in range(n_classes):
            plt.plot(fpr[i],tpr[i],label='ROC curve of class {0} (area = {1:0.2f})'''.format(i,roc_auc[i]))
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()