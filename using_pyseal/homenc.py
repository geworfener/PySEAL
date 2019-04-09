from numbers import Number
import numpy as np
import seal
from seal import ChooserEvaluator, \
    Ciphertext, \
    Decryptor, \
    Encryptor, \
    EncryptionParameters, \
    Evaluator, \
    IntegerEncoder, \
    FractionalEncoder, \
    KeyGenerator, \
    MemoryPoolHandle, \
    Plaintext, \
    SEALContext, \
    EvaluationKeys, \
    GaloisKeys, \
    PolyCRTBuilder, \
    ChooserEncoder, \
    ChooserEvaluator, \
    ChooserPoly


def initialize_fractional(
        poly_modulus_degree=4096,
        security_level_bits=128,
        plain_modulus_power_of_two=10,
        plain_modulus=None,
        encoder_integral_coefficients=1024,
        encoder_fractional_coefficients=3072,
        encoder_base=2
):
    parameters = EncryptionParameters()

    poly_modulus = "1x^" + str(poly_modulus_degree) + " + 1"
    parameters.set_poly_modulus(poly_modulus)

    if security_level_bits == 128:
        parameters.set_coeff_modulus(seal.coeff_modulus_128(poly_modulus_degree))
    elif security_level_bits == 192:
        parameters.set_coeff_modulus(seal.coeff_modulus_192(poly_modulus_degree))
    else:
        parameters.set_coeff_modulus(seal.coeff_modulus_128(poly_modulus_degree))
        print("Info: security_level_bits unknown - using default security_level_bits = 128")

    if plain_modulus is None:
        plain_modulus = 1 << plain_modulus_power_of_two

    parameters.set_plain_modulus(plain_modulus)

    context = SEALContext(parameters)

    print_parameters(context)

    global encoder
    encoder = FractionalEncoder(
        context.plain_modulus(),
        context.poly_modulus(),
        encoder_integral_coefficients,
        encoder_fractional_coefficients,
        encoder_base
    )

    keygen = KeyGenerator(context)
    public_key = keygen.public_key()
    secret_key = keygen.secret_key()

    global encryptor
    encryptor = Encryptor(context, public_key)

    global evaluator
    evaluator = Evaluator(context)

    global decryptor
    decryptor = Decryptor(context, secret_key)

    global evaluation_keys
    evaluation_keys = EvaluationKeys()

    keygen.generate_evaluation_keys(16, evaluation_keys)


class EncNum(Number):

    def __init__(self, encrypted):
        super().__init__()
        self.encrypted = encrypted

    def __add__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.add(result.encrypted, other.encrypted)
        else:
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.add_plain(result.encrypted, other_plain)
        return result

    __radd__ = __add__

    def __mul__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.multiply(result.encrypted, other.encrypted)
        else:
            other = float(other)
            if other == 0.0:
                raise ValueError("multiply_plain: plain cannot be zero")
            other_plain = encoder.encode(other)
            evaluator.multiply_plain(result.encrypted, other_plain)
        return result

    __rmul__ = __mul__

    def __sub__(self, other):
        result = EncNum(Ciphertext(self.encrypted))
        if isinstance(other, EncNum):
            evaluator.sub(result.encrypted, other.encrypted)
        else:
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.sub_plain(result.encrypted, other_plain)
        return result

    def __rsub__(self, other):
        if isinstance(other, EncNum):
            result = EncNum(Ciphertext(other.encrypted))
            evaluator.sub(result.encrypted, self.encrypted)
        else:
            result = EncNum(Ciphertext(self.encrypted))
            other = float(other)
            other_plain = encoder.encode(other)
            evaluator.sub_plain(result.encrypted, other_plain)
            evaluator.negate(result.encrypted)
        return result

    def __neg__(self):
        result = EncNum(Ciphertext(self.encrypted))
        evaluator.negate(result.encrypted)
        return result


def encrypt(n):
    plain = encoder.encode(n)
    encrypted = Ciphertext()
    encryptor.encrypt(plain, encrypted)
    return EncNum(encrypted)


encrypt_ndarray = np.vectorize(encrypt)


def decrypt(n):
    plain_result = Plaintext()
    decryptor.decrypt(n.encrypted, plain_result)
    return encoder.decode(plain_result)


decrypt_ndarray = np.vectorize(decrypt)


def recrypt(n):
    i_dec = decrypt(n)
    return encrypt(i_dec)


recrypt_ndarray = np.vectorize(recrypt)


def get_noise_budget(n):
    return decryptor.invariant_noise_budget(n.encrypted)


def print_parameters(context):
    print("/ Encryption parameters:")
    print("| poly_modulus: " + context.poly_modulus().to_string())
    print("| coeff_modulus_size: " + str(context.total_coeff_modulus().significant_bit_count()) + " bits")
    print("| plain_modulus: " + str(context.plain_modulus().value()))
    print("| noise_standard_deviation: " + str(context.noise_standard_deviation()))
