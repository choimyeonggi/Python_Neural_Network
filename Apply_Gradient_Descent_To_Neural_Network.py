"""

GRADIENT DESCENT를 NEURAL NETWORK에 적용하기. 즉 WEIGHT MATRICES를 이 방법으로 수정하기.

"""

import numpy as np
import matplotlib.pyplot as plt
from ch03.azerates20191226c import softmax
from ch04.azerates20191227b import cross_entropy
from ch04.azerates20191227d import numerical_gradient


class SimpleNetwork:
    def __init__(self):
        np.random.seed(1230)
        self.W = np.random.randn(2, 3)  # 행이 둘, 열이 셋이면서 난수는 normal distribution에서 가져와서 만든 weigthed matrix 초기형.

    def predict(self, x):
        z = x.dot(self.W)
        y = softmax(z)
        return y

    def loss(self, x, ytrue):
        """손실함수. cross entropy"""
        ypred = self.predict(x)
        ce = cross_entropy(ypred, ytrue)
        return ce

    def gradient(self, x, ytrue):
        """
        x : input, t : answer label
        """
        fn = lambda W: self.loss(x, ytrue)  # 이는 cross entropy (ypred, ytrue)를 의미한다. 즉 참값과 현재값의 엔트로피(float으로 나타나는)이
        # print('fn : ', fn(self.W))
        return numerical_gradient(fn, self.W)  # 손실함수의 값을 최소화. 그런데 지금은 모든 원소가 0인 2*3행렬?
        # return fn(np.random.randn(100))   # lambda로 정의한 fn함수는 constant함수인가?


def minimal_loss(x, y_true, network=SimpleNetwork(), learning_rate=0.1, epoch=1000):
    print('\n\nFUNCTION : MINIMAL LOSS INITIATED')
    print('input x =', x)
    print('y_true =', y_true)
    y_pred = network.predict(x)
    print('y_predicted =', y_pred)
    print('y true - y predicted = ', np.abs(y_true - y_pred))
    print('initial loss =', network.loss(x, y_true))

    print('--------------------------------------------------')
    print(f'learning rate = {learning_rate}, epoch = {epoch}')
    print('--------------------------------------------------')
    for i in range(epoch):
        new_gradient = network.gradient(x, y_true)
        network.W -= learning_rate * new_gradient
    print(f'network gradient = \n{network.gradient(x, y_true)}')
    print(f'rescaled ratio = \n{network.gradient(x, y_true) / np.min(np.abs(network.gradient(x, y_true)))}')
    refined_loss = network.loss(x, y_true)
    print('refined loss =', refined_loss)
    print('END OF FUNCTION\n\n')
    return refined_loss


if __name__ == '__main__':
    ntw = SimpleNetwork()  # 'ntw'는 generator가 되어서 class의 init항목을 실행한다.
    # print('W :', ntw.W)

    """ x = [0.6, 0.9]일 때 y_t = [0, 0, 1]이라고하자. 이 때 y_p = ntw.predict(x) 라고하면
    """

    x = np.array([0.6, 0.9])
    y_t = np.array([0., 0., 1.])
    y_p = ntw.predict(x)
    # print('y true = ', y_t)
    # print('y predicted = ', y_p)  # [0.07085565 0.53406225 0.3950821 ] 이것을 y_t=[0, 0, 1]에 가깝게 만드는 것이 목표이다.
    # print('y true - y predicted = ', np.abs(y_t - y_p))  # [0.07085565 0.53406225 0.6049179 ]
    # print('cross entropy (y predicted, y true) = ', cross_entropy(y_p, y_t))  # 0.9286614370819835
    # print(ntw.loss(x, y_t))  # same.
    # print('network gradient = ', ntw.gradient(x, y_t))

    learning_rate = 0.1
    g1 = ntw.gradient(x, y_t)
    ntw.W -= learning_rate * g1
    # print('W =', ntw.W)
    # print('cross entropy (y predicted, y true) = ', ntw.loss(x, y_t))  # 0.9286614370819835

    # epoch = 10000
    # for i in range(epoch):
    #     ng = ntw.gradient(x, y_t)
    #     learning_rate = 0.01
    #     ntw.W -= learning_rate * ng
    # print('network gradient = ', ntw.gradient(x, y_t))
    # print('cross entropy (y predicted, y true) = ', ntw.loss(x, y_t))  # 0.9286614370819835

    """
    이와같은 방법으로 gradient descent를 반복적으로 적용하는 것을 함수로 만들 수 있을 것이다.
    
    def minimal_loss(x:array, y_true:array, network=SimpleNetwork(), learning_rate=0.1, epoch=1000):
        print('y_true =', y_true)
        y_pred = network.predict(x)
        print('y_predicted =', y_pred)
        print('y true - y predicted = ', np.abs(y_true - y_pred))
        print('initial loss =', network.loss(x, y_true))
        
        print('--------------------------------------------------')
        print(f'learning rate = {learning_rate}, epoch = {epoch}')
        print('--------------------------------------------------')
        for i in range(epoch):
            new_gradient = network.gradient(x, y_true)
            network.W -= learning_rate * new_gradient
        print('network gradient = {network.gradient(x, y_true)}')
        refined_loss = network.loss(x, y_true)
        print('refined loss =', refined_loss)
        return refined_loss
    """

    """
    epoch=100,000 * refined loss = 5.6796653284403275
    epoch=56,234 * refined loss = 5.680948398863398
    epoch=31,623 * refined loss = 5.67939258410327461329‬
    epoch=17,783 * refined loss = 5.6757256103178516555‬
    epoch=10,000 * refined loss = 5.670435357500369
    epoch=5,623 * refined loss = 5.664070084606913569‬
    epoch=3,162 * refined loss = 5.6575515896455084746‬
    epoch=1,778 * refined loss = 5.6525345905222600546
    epoch=1,000 * refined loss = 5.651814390098045
    epoch=562 * refined loss = 5.659633590210888636‬
    epoch=316 * refined loss = 5.681344951015570464
    epoch=178 * refined loss = 5.7209743254989494
    epoch=100 * refined loss = 5.7740259354553
    epoch=56 * refined loss = 5.80580241655449504
    epoch=32 * refined loss = 5.71915740305759104‬
    epoch=18 * refined loss = 5.330952515605041‬
    epoch=10 * refined loss = 4.4853549332598724
    epoch=5 * refined loss = 3.125272219841004
    epoch=3 * refined loss = 2.1807067029054273‬
    epoch=2 * refined loss = 1.5738220066744806
    epoch=1 * refined loss = 0.8539188030122308

    FUNCTION : MINIMAL LOSS INITIATED
    input x = [0.6 0.9]
    y_true = [0. 0. 1.]
    y_predicted = [0.07085565 0.53406225 0.3950821 ]
    y true - y predicted =  [0.07085565 0.53406225 0.6049179 ]
    initial loss = 0.9286614370819835

    --------------------------------------------------
    learning rate = 0.1, epoch = 100000
    --------------------------------------------------
    network gradient = 
    [[ 1.62877159e-05  1.78493049e-05 -3.41370202e-05]
     [ 2.44315732e-05  2.67739582e-05 -5.12055309e-05]]
    rescaled ratio = 
    [[ 1.          1.09587526 -2.09587523]
     [ 1.49999997  1.64381295 -3.14381288]]
    refined loss = 5.6796653284403275e-05

    --------------------------------------------------
    learning rate = 0.1, epoch = 56234
    --------------------------------------------------
    network gradient = 
    [[ 2.86556394e-05  3.20153182e-05 -6.06709559e-05]
     [ 4.29834574e-05  4.80229757e-05 -9.10064347e-05]]
    rescaled ratio = 
    [[ 1.          1.1172432  -2.11724314]
     [ 1.49999994  1.67586474 -3.17586474]]
    refined loss = 0.00010102337373943519

    --------------------------------------------------
    learning rate = 0.1, epoch = 31623
    --------------------------------------------------
    network gradient = 
    [[ 5.02921098e-05  5.75163416e-05 -1.07808449e-04]
     [ 7.54381644e-05  8.62745098e-05 -1.61712674e-04]]
    rescaled ratio = 
    [[ 1.          1.14364543 -2.14364539]
     [ 1.49999999  1.7154681  -3.21546809]]
    refined loss = 0.00017959689416258023

    --------------------------------------------------
    learning rate = 0.1, epoch = 17783
    --------------------------------------------------
    network gradient = 
    [[ 8.80028305e-05  1.03526080e-04 -1.91528911e-04]
     [ 1.32004246e-04  1.55289120e-04 -2.87293366e-04]]
    rescaled ratio = 
    [[ 1.          1.17639489 -2.17639489]
     [ 1.50000001  1.76459233 -3.26459234]]
    refined loss = 0.0003191658106235085

    --------------------------------------------------
    learning rate = 0.1, epoch = 10000
    --------------------------------------------------
    network gradient = 
    [[ 0.00015343  0.00018676 -0.00034019]
     [ 0.00023015  0.00028014 -0.00051028]]
    rescaled ratio = 
    [[ 1.          1.21720484 -2.21720484]
     [ 1.50000001  1.82580725 -3.32580725]]
    refined loss = 0.0005670435357500369

    --------------------------------------------------
    learning rate = 0.1, epoch = 5623
    --------------------------------------------------
    network gradient = 
    [[ 0.00026633  0.0003378  -0.00060414]
     [ 0.0003995   0.0005067  -0.00090621]]
    rescaled ratio = 
    [[ 1.          1.26834129 -2.26834129]
     [ 1.49999999  1.90251193 -3.40251194]]
    refined loss = 0.001007303945332903

    --------------------------------------------------
    learning rate = 0.1, epoch = 3162
    --------------------------------------------------
    network gradient = 
    [[ 0.0004598   0.00061284 -0.00107264]
     [ 0.00068971  0.00091925 -0.00160896]]
    rescaled ratio = 
    [[ 1.          1.33281971 -2.33281971]
     [ 1.5         1.99922957 -3.49922957]]
    refined loss = 0.0017892320017854233

    --------------------------------------------------
    learning rate = 0.1, epoch = 1778
    --------------------------------------------------
    network gradient = 
    [[ 0.0007887   0.00111582 -0.00190452]
     [ 0.00118305  0.00167374 -0.00285678]]
    rescaled ratio = 
    [[ 1.          1.41476727 -2.41476727]
     [ 1.5         2.1221509  -3.6221509 ]]
    refined loss = 0.0031791533130046457

    --------------------------------------------------
    learning rate = 0.1, epoch = 1000
    --------------------------------------------------
    network gradient = 
    [[ 0.00134199  0.00203959 -0.00338158]
     [ 0.00201299  0.00305939 -0.00507238]]
    rescaled ratio = 
    [[ 1.          1.51982581 -2.51982581]
     [ 1.5         2.27973871 -3.77973871]]
    refined loss = 0.005651814390098046

    --------------------------------------------------
    learning rate = 0.1, epoch = 562
    --------------------------------------------------
    network gradient = 
    [[ 0.00226336  0.00374869 -0.00601205]
     [ 0.00339504  0.00562304 -0.00901808]]
    rescaled ratio = 
    [[ 1.          1.65625316 -2.65625316]
     [ 1.5         2.48437974 -3.98437975]]
    refined loss = 0.010070522402510478

    --------------------------------------------------
    learning rate = 0.1, epoch = 316
    --------------------------------------------------
    network gradient = 
    [[ 0.00377064  0.00692038 -0.01069103]
     [ 0.00565597  0.01038058 -0.01603654]]
    rescaled ratio = 
    [[ 1.          1.8353325  -2.8353325 ]
     [ 1.5         2.75299875 -4.25299875]]
    refined loss = 0.017978939718403704

    --------------------------------------------------
    learning rate = 0.1, epoch = 178
    --------------------------------------------------
    network gradient = 
    [[ 0.00617535  0.01280228 -0.01897763]
     [ 0.00926303  0.01920343 -0.02846645]]
    rescaled ratio = 
    [[ 1.          2.0731263  -3.0731263 ]
     [ 1.5         3.10968946 -4.60968946]]
    refined loss = 0.0321403051994323

    --------------------------------------------------
    learning rate = 0.1, epoch = 100
    --------------------------------------------------
    network gradient = 
    [[ 0.00991286  0.02375015 -0.03366301]
     [ 0.01486928  0.03562523 -0.05049451]]
    rescaled ratio = 
    [[ 1.          2.39589377 -3.39589377]
     [ 1.5         3.59384066 -5.09384066]]
    refined loss = 0.057740259354553

    --------------------------------------------------
    learning rate = 0.1, epoch = 56
    --------------------------------------------------
    network gradient = 
    [[ 0.01539423  0.04369491 -0.05908913]
     [ 0.02309134  0.06554236 -0.0886337 ]]
    rescaled ratio = 
    [[ 1.          2.83839578 -3.83839578]
     [ 1.5         4.25759368 -5.75759368]]
    refined loss = 0.10367504315275884

    --------------------------------------------------
    learning rate = 0.1, epoch = 32
    --------------------------------------------------
    network gradient = 
    [[ 0.02226473  0.07593313 -0.09819786]
     [ 0.0333971   0.1138997  -0.1472968 ]]
    rescaled ratio = 
    [[ 1.          3.41046652 -4.41046652]
     [ 1.5         5.11569978 -6.61569978]]
    refined loss = 0.17872366884554972

    --------------------------------------------------
    learning rate = 0.1, epoch = 18
    --------------------------------------------------
    network gradient = 
    [[ 0.02983847  0.1239623  -0.15380078]
     [ 0.04475771  0.18594346 -0.23070117]]
    rescaled ratio = 
    [[ 1.          4.15444521 -5.15444521]
     [ 1.5         6.23166781 -7.73166781]]
    refined loss = 0.2961640286447245

    --------------------------------------------------
    learning rate = 0.1, epoch = 10
    --------------------------------------------------
    network gradient = 
    [[ 0.0360911   0.18077133 -0.21686244]
     [ 0.05413666  0.271157   -0.32529366]]
    rescaled ratio = 
    [[ 1.          5.00875034 -6.00875033]
     [ 1.5         7.5131255  -9.0131255 ]]
    refined loss = 0.44853549332598724

    --------------------------------------------------
    learning rate = 0.1, epoch = 5
    --------------------------------------------------
    network gradient = 
    [[ 0.04017232  0.23868832 -0.27886064]
     [ 0.06025848  0.35803248 -0.41829095]]
    rescaled ratio = 
    [[  1.           5.94161165  -6.94161165]
     [  1.5          8.91241747 -10.41241747]]
    refined loss = 0.6250544439682008

    --------------------------------------------------
    learning rate = 0.1, epoch = 3
    --------------------------------------------------
    network gradient = 
    [[ 0.04147743  0.26848008 -0.30995751]
     [ 0.06221614  0.40272012 -0.46493626]]
    rescaled ratio = 
    [[  1.           6.47292042  -7.47292042]
     [  1.5          9.70938063 -11.20938063]]
    refined loss = 0.7269022343018091

    --------------------------------------------------
    learning rate = 0.1, epoch = 2
    --------------------------------------------------
    network gradient = 
    [[ 0.04197446  0.28487619 -0.32685066]
     [ 0.0629617   0.42731429 -0.49027598]]
    rescaled ratio = 
    [[  1.           6.7868928   -7.7868928 ]
     [  1.5         10.18033919 -11.68033919]]
    refined loss = 0.7869110033372403

    --------------------------------------------------
    learning rate = 0.1, epoch = 1
    --------------------------------------------------
    network gradient = 
    [[ 0.04233     0.30222403 -0.34455403]
     [ 0.063495    0.45333604 -0.51683104]]
    rescaled ratio = 
    [[  1.           7.13971285  -8.13971285]
     [  1.5         10.70956927 -12.20956927]]
    refined loss = 0.8539188030122308

    END OF FUNCTION
    """

    epoch_range = [1, 2, 3, 5, 10, 18, 32, 56, 100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000, 17783, 31623, 56234,
                   100000]
    ys_range = [0.8539188030122308, 1.5738220066744806, 2.1807067029054273, 3.125272219841004, 4.4853549332598724,
                5.330952515605041, 5.71915740305759104, 5.80580241655449504, 5.7740259354553, 5.7209743254989494,
                5.681344951015570464, 5.659633590210888636, 5.651814390098045, 5.6525345905222600546,
                5.6575515896455084746, 5.664070084606913569, 5.670435357500369, 5.6757256103178516555,
                5.67939258410327461329, 5.680948398863398, 5.6796653284403275]

    plt.scatter(epoch_range, ys_range)
    # plt.savefig('epoch_refined_loss_proportion')
    plt.show()
