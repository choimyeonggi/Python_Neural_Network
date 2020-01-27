import string
import secrets
def id_generator(size=6, chars=string.ascii_uppercase + string.digits + string.ascii_lowercase):
    return ''.join(secrets.choice(chars) for _ in range(size))

import uuid

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

if __name__ == '__main__':

    iteration_length = 16
    for i in range(iteration_length):
        print(id_generator(size=iteration_length))
        print(my_random_string(string_length=iteration_length))
    """
example:
l3v6eKicqt1wNtwn
9279D2574CFB4448
xSjZ0nR6KxOAXoGs
700EE7A643BE46B1
bqEf3S9ZUG2VF3Jo
0BED524CBA834864
d5NWvu52mGM2QDgj
93E4E54B7FB24DB1
dadjCHUmsONKRTHk
2AD1C633F2264055
ixoo0uelRhLL7mqS
6CC563EB38FA4459
1Yr7HomuKdpq3Ooc
B0F61A1EB35D4F16
X90d8Nyg6Sq3BTAg
A4ACFA535FCA4ABE
vv9vEh6WZMV1yts5
83E9A3E890EF4019
qz5pEzfD1dimzxv1
8098A17F50E54DE7
0KKKdgSUmKHNZlFN
E205621FB09A4A0A
6N1l6wottQXqPpnb
7EBE7783B9824539
crqITGsh4yP1py0m
869B2E5DBDA04D81
jMvueqyqPyzTWeUx
F87E73DF6409453D
71IJuqyZPtpks0zm
975CF5A155DD4B2E
ecxPrrx0POmSnI8h
9AD2E6B585AD4225
    """
