import re
import numpy as np
import pylab
import sys
from itertools import cycle

def strfloat(z):
    return (z[0], float(z[1]))

if __name__=="__main__":

    lines = [x.strip() for x in open(sys.argv[1])]

    iterations = []
    this_iter = []
    for l in lines:
        if 'Iteration' in l:
            iterations.append(this_iter)
            this_iter = [l]
        else:
            this_iter.append(l)
    iterations.append(this_iter)

    iterations.pop(0) #remove initialization junk

    iterdic = {}
    for x in iterations:
        k = re.findall('Iteration ([0-9]+),', x[0])[0]
        if k in iterdic:
            iterdic[k].extend(x)
        else:
            iterdic[k] = x 

    names_trainloss = set([])
    names_testloss = set([])
    names_testacc = set([])
    iterscores = {}
    for x in iterdic:

        #thisit_trainloss = dict([strfloat(re.findall('(loss[@a-zA-Z0-9\/ _\-]+) = ([0-9\.e\-]+) ', y)[0]) for y in iterdic[x] if 'Train net output' in y and ': loss' in y])
        #thisit_testloss = dict([strfloat(re.findall('(loss[@a-zA-Z0-9\/ _\-]+) = ([0-9\.e\-]+) ', y)[0]) for y in iterdic[x] if 'Test net output' in y and ': loss' in y])
        #thisit_testacc = dict([strfloat(re.findall('(accuracy[@a-zA-Z0-9\/ _\-]+) = ([0-9\.e\-]+)', y)[0]) for y in iterdic[x] if 'Test net output' in y and ': accuracy' in y])

        '''
        for y in iterdic[x]:
            if 'Train net output' in y and ': loss' in y:
                print str(re.findall('loss = ([0-9].[0-9]*)', y)[0])
                #import ipdb; ipdb.set_trace()
                print str(dict([strfloat(re.findall('(loss) = ([0-9\.]+)', y)[0])]))
                thisit_trainloss = dict[(float(re.findall('(loss) = ([0-9\.]+)', y)[0])])
        '''

        thisit_trainloss = dict([strfloat(re.findall('(loss) = ([0-9\.]+)', y)[0]) for y in iterdic[x] if 'Train net output' in y and ': loss' in y])
        thisit_testloss = dict([strfloat(re.findall('(loss) = ([0-9\.]+)', y)[0]) for y in iterdic[x] if 'Test net output' in y and ': loss' in y])
        thisit_testacc = dict([strfloat(re.findall('(accuracy) = ([0-9\.]+)', y)[0]) for y in iterdic[x] if 'Test net output' in y and ': accuracy' in y])
        
        #full_train_loss = dict([strfloat(re.findall('(loss) = ([0-9\.]+)', iterdic[x][0])[0])]) if 'loss' in iterdic[x][0] else None

        #print "len(thisit_trainloss): " + str(len(thisit_trainloss))
        #print "x: " + str(iterdic[x])

        iterscores[int(x)] = {}
        if len(thisit_trainloss):
            iterscores[int(x)]['trainloss'] = thisit_trainloss
            names_trainloss = names_trainloss.union(thisit_trainloss.keys())
        #if full_train_loss != None:
        #    iterscores[int(x)]['trainloss'].update(full_train_loss)
        #    names_trainloss = names_trainloss.union(['loss'])
        if len(thisit_testloss):
            iterscores[int(x)]['testloss'] = thisit_testloss
            names_testloss = names_testloss.union(thisit_testloss.keys())
        if len(thisit_testacc):
            iterscores[int(x)]['testacc'] = thisit_testacc
            names_testacc = names_testacc.union(thisit_testacc.keys())
    names_testloss = list(names_testloss)
    names_trainloss = list(names_trainloss)
    names_testacc = list(names_testacc)

    #plots:
    linestyles = cycle(['r', 'g', 'b', 'y', 'k', 'c', 'm', 'r-+', 'g-+', 'b-+', 'y-+', 'k-+', 'c-+', 'm-+', 'r-o', 'g-o', 'b-o', 'y-o', 'k-o', 'c-o', 'm-o', 'r-^', 'g-^', 'b-^', 'y-^', 'k-^', 'c-^', 'm-^'])
    #1) test accuracy
    '''
    leg = []
    pylab.figure(1)
    pylab.ylim([0,1])
    pylab.title('Validation accuracy')
    pylab.xlabel('iteration')
    pylab.ylabel('accuracy')
    pylab.hold(1)
    for k in names_testacc:
        leg.append(k)
        X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testacc'][k]) for x in iterscores if 'testacc' in iterscores[x]]))
        order = np.argsort(X)
        X = X[order]
        Y = Y[order]
        pylab.plot(X,Y, linestyles.next())
    pylab.legend(leg)
    #2) train vs test loss
    linestyles = cycle(['rs', 'gs', 'bs', 'ys', 'ks', 'cs', 'ms', 'r+', 'g+', 'b+', 'y+', 'k+', 'c+', 'm+', 'ro', 'go', 'bo', 'yo', 'ko', 'co', 'mo', 'r^', 'g^', 'b^', 'y^', 'k^', 'c^', 'm^'])
    sideplots = int(np.ceil(np.sqrt(len(names_testloss)+1)))
    f, axarray = pylab.subplots(sideplots, sideplots)
    f.suptitle('train/val loss')
    for j in range(sideplots):
        for i in range(sideplots):
            if i+(j*sideplots) >= len(names_testloss):
                continue
            k = names_testloss[i+(j*sideplots)]
            leg = []
            leg.append('train '+k)
            X, Y = map(np.array, zip(*[(int(x), iterscores[x]['trainloss'][k]) for x in iterscores if 'trainloss' in iterscores[x] and iterscores[x]['trainloss'][k]!=0]))
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            axarray[j][i].plot(X/1000.,Y, linestyles.next())
            leg.append('test '+k)
            X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testloss'][k]) for x in iterscores if 'testloss' in iterscores[x] and iterscores[x]['testloss'][k]!=0]))
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            axarray[j][i].plot(X/1000.,Y, linestyles.next())
            axarray[j][i].legend(leg)
            axarray[j][i].set_xlabel('iteration (*1000)')
            axarray[j][i].set_ylabel('loss')
    X, Y = map(np.array, zip(*[(int(x), iterscores[x]['trainloss']['loss']) for x in iterscores if 'trainloss' in iterscores[x] and 
                               'loss' in iterscores[x]['trainloss'] and iterscores[x]['trainloss']['loss']!=0]))
    order = np.argsort(X)
    X = X[order]
    Y = Y[order]
    axarray[j][i].plot(X/1000.,Y, linestyles.next())
    leg = ['joint loss']
    axarray[j][i].legend(leg)
    pylab.show()
    '''






    '''
    leg = []
    pylab.figure(1)
    pylab.ylim([0,1])
    pylab.title('Validation accuracy')
    pylab.xlabel('iteration')
    pylab.ylabel('accuracy')
    pylab.hold(1)
    for k in names_testacc:
        leg.append(k)
        X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testacc'][k]) for x in iterscores if 'testacc' in iterscores[x]]))
        order = np.argsort(X)
        X = X[order]
        Y = Y[order]
        pylab.plot(X,Y, linestyles.next())
    pylab.legend(leg)
    #2) train vs test loss
    linestyles = cycle(['rs', 'gs', 'bs', 'ys', 'ks', 'cs', 'ms', 'r+', 'g+', 'b+', 'y+', 'k+', 'c+', 'm+', 'ro', 'go', 'bo', 'yo', 'ko', 'co', 'mo', 'r^', 'g^', 'b^', 'y^', 'k^', 'c^', 'm^'])
    sideplots = int(np.ceil(np.sqrt(len(names_testloss)+1)))
    f, axarray = pylab.subplots(sideplots)
    f.suptitle('train/val loss')
    for j in range(sideplots):
        if (j*sideplots) >= len(names_testloss):
            continue
        k = names_testloss[(j*sideplots)]
        leg = []
        leg.append('train '+k)
        X, Y = map(np.array, zip(*[(int(x), iterscores[x]['trainloss'][k]) for x in iterscores if 'trainloss' in iterscores[x] and iterscores[x]['trainloss'][k]!=0]))
        order = np.argsort(X)
        X = X[order]
        Y = Y[order]
        axarray[j].plot(X/1000.,Y, linestyles.next())
        leg.append('test '+k)
        X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testloss'][k]) for x in iterscores if 'testloss' in iterscores[x] and iterscores[x]['testloss'][k]!=0]))
        order = np.argsort(X)
        X = X[order]
        Y = Y[order]
        axarray[j].plot(X/1000.,Y, linestyles.next())
        axarray[j].legend(leg)
        axarray[j].set_xlabel('iteration (*1000)')
        axarray[j].set_ylabel('loss')

    X, Y = map(np.array, zip(*[(int(x), iterscores[x]['trainloss']['loss']) for x in iterscores if 'trainloss' in iterscores[x] and 
                               'loss' in iterscores[x]['trainloss'] and iterscores[x]['trainloss']['loss']!=0]))
    order = np.argsort(X)
    X = X[order]
    Y = Y[order]
    axarray[j].plot(X/1000.,Y, linestyles.next())
    leg = ['joint loss']
    axarray[j].legend(leg)

    axarray[0].set_ylim([0,5])
    axarray[1].set_ylim([0,5])

    pylab.show()
    '''

    if len(names_testacc) > 0:

        pylab.figure(1)
        pylab.ylim([0,1])
        pylab.title('Validation accuracy')
        pylab.xlabel('iteration')
        pylab.ylabel('accuracy')
        pylab.hold(1)
        leg = []
        for k in names_testacc:
            leg.append(k)
            X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testacc'][k]) for x in iterscores if 'testacc' in iterscores[x]]))
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            pylab.plot(X,Y, linestyles.next())
        pylab.legend(leg)

    #2) train vs test loss
    if len(names_trainloss) > 0 or len(names_testloss) > 0:
        pylab.figure(2)
        pylab.clf()
        #pylab.plot(X/1000.,Y, linestyles.next())
        pylab.title('train/val loss')
        #pylab.xlabel('iteration (*1000)')
        pylab.xlabel('iteration')
        pylab.ylabel('loss')

        #k = names_testloss[(0)]
        k = 'loss'
        leg = []

        for k in names_trainloss:
            #TRAIN SET
            leg.append('train '+k)
            X, Y = map(np.array, zip(*[(int(x), iterscores[x]['trainloss'][k]) for x in iterscores if 'trainloss' in iterscores[x] and iterscores[x]['trainloss'][k]!=0]))
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            #pylab.plot(X/1000.,Y, linestyles.next())
            pylab.plot(X,Y, linestyles.next())

        for k in names_testloss:
            #TEST SET
            leg.append('test '+k)
            X, Y = map(np.array, zip(*[(int(x), iterscores[x]['testloss'][k]) for x in iterscores if 'testloss' in iterscores[x] and iterscores[x]['testloss'][k]!=0]))
            order = np.argsort(X)
            X = X[order]
            Y = Y[order]
            #pylab.plot(X/1000.,Y, linestyles.next())
            pylab.plot(X,Y, linestyles.next())
        
        pylab.legend(leg)
        pylab.ylim([0,5]) # SOFTMAX
        pylab.ylim([0,0.5]) # CONTRASTIVE

    if len(names_testacc) > 0 or len(names_trainloss) > 0 or len(names_testloss) > 0:

        pylab.show()




