"""
hybrid detector
"""

def hybrid_detector(model, data, alpha=0.01):
    """
    combing result in model and data together
    model: residuals, batch × {0, 1}
    data: probabilities, batch × [0, 1]
    alpha: adjustment factor

    judge = (P(f=1|d) * P(r|f=1)) / (P(f=0|d) * P(r|f=0))
    cpt for mbd:#[ 0.99819285  0.00180715  0.1041675   0.8958325 ]
    """
    #      P(r=1|f=1)      P(r=0|f=1)      P(r=1|f=0)      P(r=0|f=0)
    cpt = [0.99819285,  0.00180715,  0.1041675,   0.8958325]
    f = []
    for r, p in zip(model, data):
        pf1d = shrink(p, alpha)
        pf0d = 1 - pf1d
        if r > 0.5:#means r==1, P(r=1|f=1), P(r=1|f=0)
            prf1 = shrink(cpt[0], alpha)
            prf0 = shrink(cpt[2], alpha)
        else:#means r==0, P(r=0|f=1), P(r=0|f=0)
            prf1 = shrink(cpt[1], alpha)
            prf0 = shrink(cpt[3], alpha)
        judge = (pf1d * prf1) / (pf0d * prf0)
        f.append(1 if judge > 1 else 0)
    return f

def shrink(p, alpha):
    """
    shrink probability
    p:[0, 1]
    alpha:(0, 0.5)
    """
    return (1-2*alpha)*p + alpha

def shrink_cpt(cpt, alpha):
    """
    shrink a conditional probability table
    """
    cpt2 = [0] * len(cpt)
    for i in range(len(cpt)):
        cpt2[i] = shrink(cpt[i], alpha)
    return cpt2
