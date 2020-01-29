import string, secrets

def id_generator(size, chars=string.ascii_uppercase, punctuation=False, lower_case=False):
    """
    Generates a random sequence of string, that are consisted of upper/lower/number/punctuation.
    parametres:
    size : the length of the random string. only natural number.
    chars : string set for creating random strings. Do not modify (recommended).
    punctuation : enables punctuation characters >> !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~. default=False
    lower_case : enables lower letters.
    """
    if punctuation:
        chars += string.punctuation
    if lower_case:
        chars += string.ascii_lowercase
    chars += string.digits * int(round(len(chars)/len(string.digits),0))
    return ''.join(secrets.choice(chars) for _ in range(size)).replace(" ",secrets.choice(chars))  # Since punctuation contains whitespace " ", we must substitute it as another one.

"""
import uuid

def my_random_string(string_length=10):
    comment = 'Returns a random string of length string_length.'
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    #random = random.upper() # Make all characters uppercase. Maybe we don't need this.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string. -> this method returns always only upper or lower case + numbers. Hence, password security is releatively weak than id_generator.
"""

if __name__ == '__main__':

#these random generators ignore numpy.random.seed().

    iteration_length = 16
    password_size = 16
    for i in range(iteration_length):
        print(id_generator(size=password_size, punctuation=True, lower_case=True))
        """
    example:
    ;-na5e}4169T16D9
    WLx^0WTANA$M1B3I
    8562`J?51j]4M2ZE
    [9>87E(w538B)9!0
    2;3(>E)e9F5,5047
    5Y~^933H25<07576
    +968#050U8G2j20e
    7Y0RExUJ310,83R0
    ?D6E)55m_ZE2*e3p
    G}38a+9>f692]q?o
    13@0]0T1i6627p!6
    aj32T96w9i=z0M33
    OeK{94`kL9[3z51h
    ;53gQ34R=T>d\j;t
    G2X3900`{<2I9457
    06*IC.d"0g`jY9^7
        """
