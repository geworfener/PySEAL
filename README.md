**About**

This repo demonstrates and eases the usage of (Py)SEAL (https://github.com/Lab41/PySEAL). 

Classic encryption schemes try to restrict access to data.
Machine learning, on the other hand, requires the extraction of as much meaningful data as possible.
Has homomorphic encryption evolved far enough to bring these two worlds together?

Homomorphic encryption provides the ability to compute on data while the data is encrypted. In this way, the data stays safe and at the same time can be used for various applications.
`binlogreg.py` shows the efficiency of the free SEAL encryption library by solving a binary linear classification problem using logistic regression on the standard IRIS dataset. 

Moreover, to get started easily, `homenc.py` provides
- a builder for the FractionalEncoder (to initialize (Py)SEAL),
- helper functions for encoding/encrypting/decrypting numpy arrays and
- an encrypted numerical data type with overloaded mathematical operators.


**Installation**
1. Install docker.
2. Run `build-docker.sh`.
3. Run `run-docker.sh` (to run the tests in `test_homenc.py` in order to verify your working installation).


**Usage example / How to perform homomorphic calculations**
- Easy initialization of (Py)SEAL via builder method initialize_fractional().
- Easy encoding/encryption/decryption (of scalars and numpy arrays) via helper methods.
- Easy math operations with encrypted data type via overloaded mathematical operators.

```
from using_pyseal import homenc as he

he.initialize_fractional()

x = 12
y = 4
x_enc = he.encrypt(x)
y_enc = he.encrypt(y)

r = x + y
r_enc = x_enc + y_enc
r_dec = he.decrypt(r_enc)

assertEqual(r, r_dec)
```

For further examples see `test_homenc.py` and `binlogreg.py`.


**TODOs**

- Add builder for IntegerEncoder
- Add further math operators (exponentiation)
- Optimize via batching
- Optimize via relinearization
- Replace recryption with bootstrapping
- Upgrade PySEAL to SEAL 3
