"""
some utilities
"""

def number2vector(num, n):
    """
    convert a int number into a n bit vector
    """
    str_vec = bin(num).replace('0b','').zfill(n)
    vec = np.array([int(i) for i in str_vec])
    return vec

def vector2number(vec):
    """
    convert a vector to number
    """
    str_vec = [str(i) for i in vec]
    str_bin_num = ''.join(str_vec)
    num = int(str_bin_num, 2)
    return num
