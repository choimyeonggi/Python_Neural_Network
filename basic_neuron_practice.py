"""
BASIC NEURON PRACTICE

2020 01 10 JJH
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))


def relu(x, threshold=0.):
    if x > threshold:
        y = x
    else:
        y = 0
    return y


def tanh(x):
    return np.tanh(x)


def arctan(x):
    return np.arctan(x)

    ## Version 2


# def Basic_Sigmoid_neuron(weight_input=1, bias_input=1, expected_output=0, learning_rate=0.25, weight=np.random.randn(),
#                         bias=np.random.randn(), renew_weight=True, show_plot=False):
#    plotter = []
#    cnt = 1
#    if renew_weight:
#        bias_input = 0
#        while True:
#            Error_gap = expected_output - sigmoid(weight_input * weight + bias_input * bias)
#            print(f'{cnt}::Weight = {weight} Error gap = {Error_gap}')
#            weight += learning_rate * weight_input * Error_gap
#            cnt += 1
#            plotter.append(Error_gap)
#            plt.title('Weight=1, Bias=0')
#            if abs(Error_gap) <= 0.01:
#                break
#    else:
#        weight_input = 0
#        while True:
#            Error_gap = expected_output - sigmoid(weight_input * weight + bias_input * bias)  # Can I simplify this part?
#            print(f'{cnt}::bias = {bias} Error gap = {Error_gap}')
#            bias += learning_rate * bias_input * Error_gap
#            cnt += 1
#            plotter.append(Error_gap)
#            plt.title('Weight=0, Bias=1')
#            if abs(Error_gap) <= 0.01:
#                break
#    if show_plot:
#        plt.plot(range(len(plotter)), plotter)
#        plt.show()
#    return plotter

## Version 3
# Now let's make a improved function which differentiates weight and bias at same time. And let's allow it to accept other activation functions not only sigmoid.
#
# def Basic_neuron(weight_input=1, bias_input=1, expected_output=0, learning_rate=0.25,
#                  activation=sigmoid,
#                  weight=np.random.randn(),
#                  bias=np.random.randn(), iteration_check=False):
#     neuron_history = []
#     weight_bias_history = [(weight, bias)]
#     cnt = 1
#     while True:
#         Error_gap = expected_output - activation(weight_input * weight + bias_input * bias)
#         if iteration_check:
#             print(f'{cnt}::Weight = {weight}, Bias = {bias} Error gap = {Error_gap}')
#         weight += learning_rate * weight_input * Error_gap
#         bias += learning_rate * bias_input * Error_gap
#         cnt += 1
#         neuron_history.append(Error_gap)
#         weight_bias_history.append((weight, bias))
#         plt.title('Weight and Bias simultaneously changed')
#         if abs(Error_gap) <= 0.01:
#             break
#     return neuron_history, weight_bias_history


""" 
The function we made first(Version 2) had an option that renew weight or bias only, add a switch for it. modify abs(Error_gap) <= 0.01 into tolerance=0.01, abs(Error_gap) <= tolerance.
So that we can handle error limit as small as we want. In fact, We don't actually need renew weight/bias only switch, because we can set weight/input = 0, bias/input = 0 to do so.
"""


## Version 4
# def Basic_neuron(weight_input=1, bias_input=1, expected_output=0, learning_rate=0.25, tolerance=0.01,
#                  activation=sigmoid,
#                  weight=np.random.randn(),
#                  bias=np.random.randn(), iteration_check=False, renew_weight_only=False,
#                  renew_bias_only=False):  # It seems, we don't need renew booleans anymore.
#     """
#     Single Neuron iterator.
#
#    e.g.
#     X = Basic_neuron(weight_input=10, bias=10, expected_output=0.5)
#     Y = Basic_neuron(iteration_check=True, tolerance=0.001)
#     Z = Basic_neuron(weight=10, learning_rate=0.01 ,renew_weight_only=True)
#
#     Based on those formulae :
#     Error_n = expected_output - activation(weight_input * weight + bias_input * bias)
#     weight_n+1 = weight_n + learning_rate * weight_input * Error_n
#     bias_n+1 = bias_n + learning_rate * bias_input * Error_n
#
#     IF iteration is not valid, in other words if previous error difference is same to current one, the iteration will be stopped forcely.
#
#     Returns Neuron_history that stores consequences of error and weight_bias_history that stores consequences of weights and biases by tuple.
#     Return (neuron_historym weight_bias_history)
#     """
#     neuron_history = []
#     weight_bias_history = [(weight, bias)]
#     cnt = 1
#
#     Previous_Comparison = None
#
#     if renew_weight_only:
#         bias_input = 0
#         bias = 0
#
#     if renew_bias_only:
#         weight_input = 0
#         weight = 0  # When switching, not only input but also weight itself. (Performance gets better)
#
#     if renew_weight_only and renew_bias_only or bias_input * bias + weight_input * weight == 0: raise ValueError(
#         'At least one of the factor must be activated')  # if weights and biases are ZERO, error renewal won't be occured. But we don't need it anymore.
#
#     # start of the iteration
#     print(
#         f'Weight Input = {weight_input}, Bias Input = {bias_input}, Expected Output = {expected_output}, Learning Rate = {learning_rate}, Tolerance = {tolerance}')
#
#     while True:
#         Error_gap = expected_output - activation(weight_input * weight + bias_input * bias)
#
#         if Previous_Comparison == Error_gap: raise ValueError(
#             'Error difference has not been changed at all. Please handle initial settings.')
#         if iteration_check: print(f'{cnt}::Weight = {weight}, Bias = {bias} Error gap = {Error_gap}')
#
#         cnt += 1
#         weight += learning_rate * weight_input * Error_gap
#         bias += learning_rate * bias_input * Error_gap
#         neuron_history.append(Error_gap)  # add gap value to this list so that we can make graph via pyplot.
#         weight_bias_history.append((weight, bias))  # same.
#
#         Previous_Comparison = Error_gap
#
#         if abs(Error_gap) <= tolerance:
#             print('\n')
#             break  # end of the iteration
#     return neuron_history, weight_bias_history

## Version 5 (Complete)


def Basic_neuron(weight_input=1, bias_input=1, expected_output=0, learning_rate=0.25, tolerance=0.01,
                 activation=sigmoid,
                 weight=np.random.randn(),
                 bias=np.random.randn(),
                 iteration_check=False):  # Improved version.
    """
    Single Neuron iterator.

    The weights and biases are initialised by random number from normal distribution, and activation function is initialised by Sigmoid.
    You can modify them as you want.

    e.g.
    X = Basic_neuron(weight_input=10, bias=10, expected_output=0.5)
    Y = Basic_neuron(iteration_check=True, tolerance=0.001)
    Z = Basic_neuron(weight=10, learning_rate=0.01 ,renew_weight_only=True)

    Based on those formulae :
    Error_n = expected_output - activation(weight_input * weight + bias_input * bias)
    weight_n+1 = weight_n + learning_rate * weight_input * Error_n
    bias_n+1 = bias_n + learning_rate * bias_input * Error_n

    IF THE ITERATION IS NOT VALID, in other words if previous error difference is equal to current one, it will be stopped immediately.

    Returns Neuron_history that stores consequences of error and weight_bias_history that stores consequences of weights and biases by tuple.
    Return (neuron_historym weight_bias_history)
    """
    neuron_history = []
    weight_bias_history = [(weight, bias)]  # lists for return.
    if iteration_check: cnt = 1

    Previous_Comparison = None

    # start the iteration
    print(
        f'Weight Input = {weight_input}, Bias Input = {bias_input}, Expected Output = {expected_output}, Learning Rate = {learning_rate}, Tolerance = {tolerance}')

    while True:  # we're not sure how many times we need to iterate it.
        Error_gap = expected_output - activation(weight_input * weight + bias_input * bias)

        if Previous_Comparison == Error_gap: raise ValueError(
            'Error difference has not been changed at all. Please handle initial settings.')
        if iteration_check: print(f'{cnt}::Weight = {weight}, Bias = {bias} Error gap = {Error_gap}')

        cnt += 1
        weight += learning_rate * weight_input * Error_gap
        bias += learning_rate * bias_input * Error_gap

        neuron_history.append(Error_gap)  # add gap value to this list so that we can make graph via pyplot.
        weight_bias_history.append((weight, bias))  # same.

        Previous_Comparison = Error_gap  # Set Previous_Comparison to Error_gap for valid iteration.

        if abs(Error_gap) <= tolerance:  # if the error gap is at limit(Condition for stop iteration)
            print('\n')
            break  # then stop it(this part always works).
    return neuron_history, weight_bias_history


if __name__ == '__main__':
    np.random.seed(110)
    ## Version 1
    # initial_input = 1
    # expected_output = 0
    # lr = 0.25
    # weight = -0.12568
    # plotter = []
    # cnt = 1
    # while True:  # Since We don't know how many time we need to iterate
    #     Error_gap = expected_output - sigmoid(initial_input*weight)  # the error is defined by difference between expectation and real result.
    #     print(f'{cnt}::Weight ={weight} Error gap ={Error_gap}')
    #     weight += lr * initial_input * Error_gap
    #     cnt += 1
    #     plotter.append(Error_gap)
    #     if abs(Error_gap) <= 0.01:
    #         break
    # plt.plot(range(len(plotter)), plotter)
    # plt.show()
    # X = Basic_neuron(iteration_check=True, renew_bias_only=True, renew_weight_only=True)  # Error.
    # X = Basic_neuron(iteration_check=True, renew_bias_only=True)
    # Y = Basic_neuron(iteration_check=True, renew_weight_only=True)
    # A = Basic_neuron(iteration_check=True, renew_weight_only=True, weight=10)
    # B = Basic_neuron(iteration_check=True, activation=relu)
    # A = Basic_neuron(iteration_check=True, activation=tanh, renew_weight_only=True, weight=10)
    # A = Basic_neuron(iteration_check=True, activation=arctan, bias=500, renew_bias_only=True)
    A = Basic_neuron(iteration_check=True)

    """ 
    A = Basic_neuron(iteration_check=True) result.
    Weight Input = 1, Bias Input = 1, Expected Output = 0, Learning Rate = 0.25, Tolerance = 0.01
    1::Weight = -1.0567893931999632, Bias = -0.6591004915527334 Error gap = -0.1524013302913216
    2::Weight = -1.0948897257727936, Bias = -0.6972008241255638 Error gap = -0.14281660716033834
    3::Weight = -1.1305938775628783, Bias = -0.7329049759156484 Error gap = -0.13429575309185782
    4::Weight = -1.1641678158358428, Bias = -0.7664789141886128 Error gap = -0.12667901417101043
    5::Weight = -1.1958375693785954, Bias = -0.7981486677313654 Error gap = -0.11983577589256067
    6::Weight = -1.2257965133517357, Bias = -0.8281076117045055 Error gap = -0.11365848518509224
    7::Weight = -1.2542111346480087, Bias = -0.8565222330007786 Error gap = -0.10805796341236368
    8::Weight = -1.2812256255010996, Bias = -0.8835367238538695 Error gap = -0.10295977293593778
    9::Weight = -1.306965568735084, Bias = -0.9092766670878539 Error gap = -0.09830138341500927
    10::Weight = -1.3315409145888364, Bias = -0.9338520129416062 Error gap = -0.09402994761705523
    11::Weight = -1.3550484014931001, Bias = -0.9573594998458701 Error gap = -0.0901005441495101
    12::Weight = -1.3775735375304776, Bias = -0.9798846358832476 Error gap = -0.08647477995195736
    13::Weight = -1.399192232518467, Bias = -1.0015033308712369 Error gap = -0.08311967168391104
    14::Weight = -1.4199721504394447, Bias = -1.0222832487922147 Error gap = -0.08000674467930226
    15::Weight = -1.4399738366092703, Bias = -1.0422849349620402 Error gap = -0.07711130269548801
    16::Weight = -1.4592516622831422, Bias = -1.0615627606359122 Error gap = -0.07441183257622305
    17::Weight = -1.4778546204271978, Bias = -1.0801657187799678 Error gap = -0.0718895161375657
    18::Weight = -1.4958269994615894, Bias = -1.0981380978143593 Error gap = -0.06952782777672027
    19::Weight = -1.5132089564057694, Bias = -1.1155200547585393 Error gap = -0.0673122010105465
    20::Weight = -1.530037006658406, Bias = -1.1323481050111759 Error gap = -0.06522975074944763
    21::Weight = -1.546344444345768, Bias = -1.1486555426985379 Error gap = -0.06326904088029306
    22::Weight = -1.5621617045658411, Bias = -1.164472802918611 Error gap = -0.06141988887311252
    23::Weight = -1.5775166767841193, Bias = -1.1798277751368893 Error gap = -0.05967320079191047
    24::Weight = -1.592434976982097, Bias = -1.1947460753348669 Error gap = -0.058020831392895676
    25::Weight = -1.6069401848303209, Bias = -1.2092512831830908 Error gap = -0.05645546501820631
    26::Weight = -1.6210540510848723, Bias = -1.2233651494376425 Error gap = -0.05497051380348581
    27::Weight = -1.6347966795357438, Bias = -1.237107777888514 Error gap = -0.05356003036161204
    28::Weight = -1.6481866871261468, Bias = -1.250497785478917 Error gap = -0.0522186326191961
    29::Weight = -1.6612413452809458, Bias = -1.263552443633716 Error gap = -0.05094143889521335
    30::Weight = -1.6739767050047492, Bias = -1.2762878033575193 Error gap = -0.04972401164391271
    31::Weight = -1.6864077079157274, Bias = -1.2887188062684976 Error gap = -0.04856230855366445
    32::Weight = -1.6985482850541436, Bias = -1.3008593834069138 Error gap = -0.04745263991263071
    33::Weight = -1.7104114450323014, Bias = -1.3127225433850715 Error gap = -0.04639163133120822
    34::Weight = -1.7220093528651035, Bias = -1.3243204512178737 Error gap = -0.04537619105804705
    35::Weight = -1.7333534006296152, Bias = -1.3356644989823854 Error gap = -0.04440348124735973
    36::Weight = -1.7444542709414552, Bias = -1.3467653692942254 Error gap = -0.04347089263515987
    37::Weight = -1.7553219941002451, Bias = -1.3576330924530153 Error gap = -0.04257602216495279
    38::Weight = -1.7659659996414834, Bias = -1.3682770979942536 Error gap = -0.04171665317239092
    39::Weight = -1.776395162934581, Bias = -1.3787062612873513 Error gap = -0.040890737796029246
    40::Weight = -1.7866178473835883, Bias = -1.3889289457363585 Error gap = -0.040096381329600986
    41::Weight = -1.7966419427159885, Bias = -1.3989530410687587 Error gap = -0.03933182827182458
    42::Weight = -1.8064748997839446, Bias = -1.4087859981367148 Error gap = -0.03859544986398006
    43::Weight = -1.8161237622499395, Bias = -1.4184348606027097 Error gap = -0.037885732934439705
    44::Weight = -1.8255951954835494, Bias = -1.4279062938363196 Error gap = -0.03720126989389111
    45::Weight = -1.8348955129570221, Bias = -1.4372066113097923 Error gap = -0.036540749745873013
    46::Weight = -1.8440307003934904, Bias = -1.4463417987462606 Error gap = -0.03590294999505531
    47::Weight = -1.8530064378922542, Bias = -1.4553175362450244 Error gap = -0.03528672935092348
    48::Weight = -1.8618281202299851, Bias = -1.4641392185827553 Error gap = -0.03469102113758344
    49::Weight = -1.870500875514381, Bias = -1.4728119738671512 Error gap = -0.03411482733162267
    50::Weight = -1.8790295823472867, Bias = -1.4813406807000569 Error gap = -0.03355721315962789
    51::Weight = -1.8874188856371936, Bias = -1.4897299839899638 Error gap = -0.033017302195305236
    52::Weight = -1.89567321118602, Bias = -1.4979843095387901 Error gap = -0.03249427190337003
    53::Weight = -1.9037967791618624, Bias = -1.5061078775146326 Error gap = -0.031987349583636415
    54::Weight = -1.9117936165577716, Bias = -1.5141047149105418 Error gap = -0.031495808674180364
    55::Weight = -1.9196675687263167, Bias = -1.521978667079087 Error gap = -0.03101896537719039
    56::Weight = -1.9274223100706143, Bias = -1.5297334084233845 Error gap = -0.030556175575256677
    57::Weight = -1.9350613539644284, Bias = -1.5373724523171985 Error gap = -0.030106832009467115
    58::Weight = -1.9425880619667952, Bias = -1.5448991603195654 Error gap = -0.029670361693846816
    59::Weight = -1.950005652390257, Bias = -1.552316750743027 Error gap = -0.02924622354345906
    60::Weight = -1.9573172082761217, Bias = -1.5596283066288918 Error gap = -0.028833906195930237
    61::Weight = -1.9645256848251043, Bias = -1.5668367831778744 Error gap = -0.02843292600831527
    62::Weight = -1.971633916327183, Bias = -1.5739450146799532 Error gap = -0.028042825213119637
    63::Weight = -1.9786446226304628, Bias = -1.580955720983233 Error gap = -0.02766317021897313
    64::Weight = -1.985560415185206, Bias = -1.5878715135379762 Error gap = -0.027293550042936765
    65::Weight = -1.9923838026959402, Bias = -1.5946949010487104 Error gap = -0.026933574862741873
    66::Weight = -1.9991171964116257, Bias = -1.6014282947643959 Error gap = -0.026582874678429992
    67::Weight = -2.0057629150812333, Bias = -1.6080740134340032 Error gap = -0.026241098073902982
    68::Weight = -2.012323189599709, Bias = -1.614634287952479 Error gap = -0.025907911069818726
    69::Weight = -2.018800167367164, Bias = -1.6211112657199338 Error gap = -0.02558299606009456
    70::Weight = -2.0251959163821875, Bias = -1.6275070147349575 Error gap = -0.02526605082501795
    71::Weight = -2.031512429088442, Bias = -1.633823527441212 Error gap = -0.024956787614624276
    72::Weight = -2.037751625992098, Bias = -1.6400627243448682 Error gap = -0.024654932296592134
    73::Weight = -2.0439153590662458, Bias = -1.6462264574190162 Error gap = -0.02436022356343678
    74::Weight = -2.0500054149571048, Bias = -1.6523165133098754 Error gap = -0.024072412194257632
    75::Weight = -2.056023518005669, Bias = -1.6583346163584398 Error gap = -0.02379126036672365
    76::Weight = -2.06197133309735, Bias = -1.6642824314501208 Error gap = -0.023516541015365166
    77::Weight = -2.067850468351191, Bias = -1.670161566703962 Error gap = -0.02324803723258733
    78::Weight = -2.073662477659338, Bias = -1.6759735760121088 Error gap = -0.022985541709133375
    79::Weight = -2.079408863086621, Bias = -1.6817199614393923 Error gap = -0.022728856211008362
    80::Weight = -2.0850910771393734, Bias = -1.6874021754921444 Error gap = -0.022477791090129276
    81::Weight = -2.0907105249119056, Bias = -1.6930216232646769 Error gap = -0.022232164826198896
    82::Weight = -2.0962685661184555, Bias = -1.6985796644712265 Error gap = -0.021991803597510105
    83::Weight = -2.101766517017833, Bias = -1.704077615370604 Error gap = -0.021756540878577254
    84::Weight = -2.1072056522374774, Bias = -1.7095167505902484 Error gap = -0.02152621706266406
    85::Weight = -2.1125872065031435, Bias = -1.7148983048559145 Error gap = -0.021300679107433853
    86::Weight = -2.117912376280002, Bias = -1.720223474632773 Error gap = -0.021079780202091006
    87::Weight = -2.1231823213305248, Bias = -1.7254934196832956 Error gap = -0.02086337945451209
    88::Weight = -2.128398166194153, Bias = -1.7307092645469235 Error gap = -0.020651341596983768
    89::Weight = -2.133561001593399, Bias = -1.7358720999461694 Error gap = -0.020443536709272504
    90::Weight = -2.138671885770717, Bias = -1.7409829841234876 Error gap = -0.020239839957849618
    91::Weight = -2.1437318457601795, Bias = -1.7460429441129501 Error gap = -0.020040131350185675
    92::Weight = -2.148741878597726, Bias = -1.7510529769504966 Error gap = -0.019844295503110395
    93::Weight = -2.1537029524735036, Bias = -1.756014050826274 Error gap = -0.019652221424309998
    94::Weight = -2.1586160078295813, Bias = -1.7609271061823515 Error gap = -0.019463802306102906
    95::Weight = -2.163481958406107, Bias = -1.7657930567588773 Error gap = -0.019278935330698275
    96::Weight = -2.1683016922387814, Bias = -1.7706127905915519 Error gap = -0.019097521486200128
    97::Weight = -2.1730760726103315, Bias = -1.775387170963102 Error gap = -0.0189194653926732
    98::Weight = -2.1778059389585, Bias = -1.7801170373112702 Error gap = -0.01874467513763592
    99::Weight = -2.182492107742909, Bias = -1.7848032060956793 Error gap = -0.018573062120391115
    100::Weight = -2.1871353732730068, Bias = -1.7894464716257772 Error gap = -0.018404540904646785
    101::Weight = -2.1917365084991687, Bias = -1.7940476068519389 Error gap = -0.018239029078917446
    102::Weight = -2.1962962657688982, Bias = -1.7986073641216682 Error gap = -0.01807644712423221
    103::Weight = -2.200815377549956, Bias = -1.8031264759027263 Error gap = -0.01791671828870822
    104::Weight = -2.205294557122133, Bias = -1.8076056554749034 Error gap = -0.017759768468578206
    105::Weight = -2.2097344992392776, Bias = -1.812045597592048 Error gap = -0.01760552609528904
    106::Weight = -2.2141358807631, Bias = -1.8164469791158702 Error gap = -0.017453922028313276
    107::Weight = -2.218499361270178, Bias = -1.8208104596229484 Error gap = -0.01730488945334049
    108::Weight = -2.2228255836335133, Bias = -1.8251366819862835 Error gap = -0.01715836378553624
    109::Weight = -2.2271151745798976, Bias = -1.8294262729326676 Error gap = -0.017014282577577547
    110::Weight = -2.231368745224292, Bias = -1.833679843577062 Error gap = -0.01687258543219278
    111::Weight = -2.23558689158234, Bias = -1.8378979899351102 Error gap = -0.016733213918950718
    112::Weight = -2.2397701950620776, Bias = -1.8420812934148478 Error gap = -0.016596111495060774
    113::Weight = -2.2439192229358427, Bias = -1.846230321288613 Error gap = -0.01646122342996085
    114::Weight = -2.2480345287933328, Bias = -1.8503456271461034 Error gap = -0.01632849673348365
    115::Weight = -2.252116652976704, Bias = -1.8544277513294742 Error gap = -0.01619788008740519
    116::Weight = -2.2561661229985552, Bias = -1.8584772213513254 Error gap = -0.016069323780191727
    117::Weight = -2.260183453943603, Bias = -1.8624945522963734 Error gap = -0.015942779644772254
    118::Weight = -2.2641691488547964, Bias = -1.8664802472075666 Error gap = -0.01581820099917445
    119::Weight = -2.26812369910459, Bias = -1.87043479745736 Error gap = -0.015695542589872175
    120::Weight = -2.272047584752058, Bias = -1.8743586831048282 Error gap = -0.01557476053770075
    121::Weight = -2.2759412748864833, Bias = -1.8782523732392533 Error gap = -0.015455812286206107
    122::Weight = -2.2798052279580348, Bias = -1.882116326310805 Error gap = -0.015338656552300968
    123::Weight = -2.28363989209611, Bias = -1.8859509904488803 Error gap = -0.015223253279108874
    124::Weight = -2.287445705415887, Bias = -1.8897568037686574 Error gap = -0.015109563590884
    125::Weight = -2.2912230963136078, Bias = -1.8935341946663784 Error gap = -0.014997549749901242
    126::Weight = -2.294972483751083, Bias = -1.8972835821038536 Error gap = -0.014887175115216776
    127::Weight = -2.2986942775298873, Bias = -1.9010053758826577 Error gap = -0.014778404103205503
    128::Weight = -2.3023888785556887, Bias = -1.9046999769084592 Error gap = -0.01467120214978662
    129::Weight = -2.3060566790931354, Bias = -1.9083677774459058 Error gap = -0.014565535674254023
    130::Weight = -2.309698063011699, Bias = -1.9120091613644694 Error gap = -0.014461372044632527
    131::Weight = -2.313313406022857, Bias = -1.9156245043756275 Error gap = -0.014358679544485603
    132::Weight = -2.3169030759089786, Bias = -1.9192141742617488 Error gap = -0.014257427341104155
    133::Weight = -2.3204674327442545, Bias = -1.922778531097025 Error gap = -0.014157585455009964
    134::Weight = -2.324006829108007, Bias = -1.9263179274607773 Error gap = -0.014059124730711024
    135::Weight = -2.3275216102906846, Bias = -1.929832708643455 Error gap = -0.013962016808649058
    136::Weight = -2.331012114492847, Bias = -1.9333232128456173 Error gap = -0.013866234098283308
    137::Weight = -2.334478673017418, Bias = -1.936789771370188 Error gap = -0.013771749752257173
    138::Weight = -2.3379216104554823, Bias = -1.9402327088082523 Error gap = -0.013678537641597256
    139::Weight = -2.3413412448658817, Bias = -1.9436523432186517 Error gap = -0.013586572331897263
    140::Weight = -2.344737887948856, Bias = -1.947048986301626 Error gap = -0.013495829060441309
    141::Weight = -2.348111845213966, Bias = -1.9504229435667364 Error gap = -0.013406283714223864
    142::Weight = -2.351463416142522, Bias = -1.9537745144952923 Error gap = -0.013317912808825648
    143::Weight = -2.3547928943447283, Bias = -1.9571039926974987 Error gap = -0.013230693468106803
    144::Weight = -2.358100567711755, Bias = -1.9604116660645254 Error gap = -0.01314460340468103
    145::Weight = -2.361386718562925, Bias = -1.9636978169156958 Error gap = -0.013059620901135481
    146::Weight = -2.364651623788209, Bias = -1.9669627221409796 Error gap = -0.012975724791963903
    147::Weight = -2.3678955549862, Bias = -1.9702066533389706 Error gap = -0.012892894446181316
    148::Weight = -2.3711187785977454, Bias = -1.9734298769505159 Error gap = -0.01281110975059069
    149::Weight = -2.374321556035393, Bias = -1.9766326543881636 Error gap = -0.012730351093673223
    150::Weight = -2.377504143808811, Bias = -1.979815242161582 Error gap = -0.012650599350075276
    151::Weight = -2.38066679364633, Bias = -1.9829778919991008 Error gap = -0.012571835865666492
    152::Weight = -2.3838097526127466, Bias = -1.9861208509655175 Error gap = -0.012494042443144683
    153::Weight = -2.3869332632235327, Bias = -1.9892443615763036 Error gap = -0.012417201328164257
    154::Weight = -2.3900375635555737, Bias = -1.9923486619083446 Error gap = -0.012341295195966285
    155::Weight = -2.3931228873545654, Bias = -1.9954339857073362 Error gap = -0.012266307138489065
    156::Weight = -2.3961894641391877, Bias = -1.9985005624919585 Error gap = -0.012192220651939265
    157::Weight = -2.3992375193021727, Bias = -2.0015486176549433 Error gap = -0.012119019624804543
    158::Weight = -2.402267274208374, Bias = -2.0045783725611446 Error gap = -0.01204668832628952
    159::Weight = -2.4052789462899464, Bias = -2.007590044642717 Error gap = -0.011975211395157738
    160::Weight = -2.4082727491387357, Bias = -2.0105838474915063 Error gap = -0.011904573828963162
    161::Weight = -2.4112488925959763, Bias = -2.013559990948747 Error gap = -0.011834760973655389
    162::Weight = -2.4142075828393903, Bias = -2.016518681192161 Error gap = -0.011765758513543636
    163::Weight = -2.4171490224677763, Bias = -2.019460120820547 Error gap = -0.011697552461605108
    164::Weight = -2.4200734105831776, Bias = -2.0223845089359482 Error gap = -0.011630129150124029
    165::Weight = -2.4229809428707085, Bias = -2.025292041223479 Error gap = -0.011563475221648334
    166::Weight = -2.4258718116761204, Bias = -2.028182910028891 Error gap = -0.011497577620251484
    167::Weight = -2.428746206081183, Bias = -2.0310573044339537 Error gap = -0.011432423583087511
    168::Weight = -2.431604311976955, Bias = -2.0339154103297257 Error gap = -0.011368000632227837
    169::Weight = -2.434446312135012, Bias = -2.0367574104877826 Error gap = -0.011304296566769083
    170::Weight = -2.437272386276704, Bias = -2.0395834846294747 Error gap = -0.01124129945520128
    171::Weight = -2.4400827111405046, Bias = -2.042393809493275 Error gap = -0.011178997628026668
    172::Weight = -2.4428774605475114, Bias = -2.045188558900282 Error gap = -0.01111737967061953
    173::Weight = -2.445656805465166, Bias = -2.047967903817937 Error gap = -0.011056434416317838
    174::Weight = -2.4484209140692457, Bias = -2.0507320124220163 Error gap = -0.010996150939738087
    175::Weight = -2.45116995180418, Bias = -2.0534810501569507 Error gap = -0.010936518550304973
    176::Weight = -2.453904081441756, Bias = -2.0562151797945267 Error gap = -0.010877526785987796
    177::Weight = -2.456623463138253, Bias = -2.0589345614910237 Error gap = -0.010819165407236087
    178::Weight = -2.459328254490062, Bias = -2.0616393528428327 Error gap = -0.010761424391106984
    179::Weight = -2.462018610587839, Bias = -2.0643297089406096 Error gap = -0.010704293925577403
    180::Weight = -2.4646946840692334, Bias = -2.067005782422004 Error gap = -0.01064776440403425
    181::Weight = -2.467356625170242, Bias = -2.0696677235230125 Error gap = -0.01059182641993618
    182::Weight = -2.470004581775226, Bias = -2.0723156801279967 Error gap = -0.010536470761640769
    183::Weight = -2.4726386994656364, Bias = -2.074949797818407 Error gap = -0.010481688407391112
    184::Weight = -2.4752591215674844, Bias = -2.077570219920255 Error gap = -0.010427470520456167
    185::Weight = -2.4778659891975985, Bias = -2.080177087550369 Error gap = -0.010373808444419381
    186::Weight = -2.4804594413087035, Bias = -2.082770539661474 Error gap = -0.010320693698610342
    187::Weight = -2.483039614733356, Bias = -2.0853507130861266 Error gap = -0.010268117973674445
    188::Weight = -2.485606644226775, Bias = -2.0879177425795454 Error gap = -0.010216073127275682
    189::Weight = -2.488160662508594, Bias = -2.0904717608613645 Error gap = -0.010164551179928032
    190::Weight = -2.490701800303576, Bias = -2.0930128986563465 Error gap = -0.010113544310950847
    191::Weight = -2.4932301863813136, Bias = -2.095541284734084 Error gap = -0.01006304485454405
    192::Weight = -2.4957459475949495, Bias = -2.09805704594772 Error gap = -0.010013045295979007
    193::Weight = -2.4982492089189443, Bias = -2.100560307271715 Error gap = -0.009963538267901115
    """

    """
    Since Basic_neuron function returns neuron_history, which stores error gaps, and weight_bias_history which stores differentiation of weights and biases,
    return type is essentially tuple.Hence Basic_neuron()[0] is neuron_history, and [1] is weight_bias_history.

    so the minimal unit for graph is like below.

    plt.plot(range(len(A[0])), A[0])
     -> plt.show().  (neuron_history)

    And weight_bias_history is,
    split_plot_x = []
    split_plot_y = []
    for x in range(len(A[1])):
        print(A[1][x][0])
        split_plot_x.append(A[1][x][0])  # list comprehension : split_plot_x = [A[1][i][0] for i in range(len(A[1]))]
        split_plot_y.append(A[1][x][1])  # list comprehension : split_plot_y = [A[1][j][1] for j in range(len(A[1]))]

    plt.scatter(split_plot_x, split_plot_y)
    plt.show()
    """
    # Neuron History Graph
    plt.plot(range(len(A[0])), A[0])
    plt.show()

    # Weight Bias History Graph
    split_plot_x = []
    split_plot_y = []
    for x in range(len(A[1])):
        print(A[1][x][0])
        split_plot_x.append(A[1][x][0])  # list comprehension : split_plot_x = [A[1][i][0] for i in range(len(A[1]))]
        split_plot_y.append(A[1][x][1])  # list comprehension : split_plot_y = [A[1][j][1] for j in range(len(A[1]))]

    plt.scatter(split_plot_x, split_plot_y)
    plt.show()
