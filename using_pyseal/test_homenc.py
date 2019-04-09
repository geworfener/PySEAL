import unittest
import numpy as np
from using_pyseal import homenc as he

PRECISION = 100
ARRAY_SIZE = 10
RANGE_FROM = -100000
RANGE_TO = 100000


class TestScalarOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        he.initialize_fractional()

    def setUp(self):
        self.x = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.y = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.x_enc = he.encrypt(self.x)
        self.y_enc = he.encrypt(self.y)

    def test_add_scalar(self):
        r = self.x + self.y
        r_enc = self.x_enc + self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_mul_scalar(self):
        r = self.x * self.y
        r_enc = self.x_enc * self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_sub_scalar(self):
        r = self.x - self.y
        r_enc = self.x_enc - self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_neg_scalar(self):
        r = -self.x
        r_enc = -self.x_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_div_scalar(self):
        with self.assertRaises(TypeError):
            r_enc = self.x_enc / self.y_enc

    def test_compare_scalar(self):
        with self.assertRaises(TypeError):
            r_enc = self.x_enc < self.y_enc


class TestScalarOperationsPlain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        he.initialize_fractional()

    def setUp(self):
        self.x = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.y = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.x_enc = he.encrypt(self.x)
        self.y_enc = he.encrypt(self.y)

    def test_add_scalar(self):
        r = self.x + self.y
        r_enc = self.x_enc + self.y
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_mul_scalar(self):
        r = self.x * self.y
        r_enc = self.x_enc * self.y
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_sub_scalar(self):
        r = self.x - self.y
        r_enc = self.x_enc - self.y
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_add_scalar_r(self):
        r = self.x + self.y
        r_enc = self.x + self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_mul_scalar_r(self):
        r = self.x * self.y
        r_enc = self.x * self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_sub_scalar_r(self):
        r = self.x - self.y
        r_enc = self.x - self.y_enc
        r_dec = he.decrypt(r_enc)
        self.assertEqual(r, r_dec)

    def test_div_scalar_r(self):
        with self.assertRaises(TypeError):
            r_enc = self.x / self.y_enc

    def test_compare_scalar_r(self):
        with self.assertRaises(TypeError):
            r_enc = self.x < self.y_enc

    def test_mul_scalar_zero(self):
        with self.assertRaises(ValueError):
            r_enc = 0.0 * self.y_enc

    def test_mul_scalar_zero_r(self):
        with self.assertRaises(ValueError):
            r_enc = self.x_enc * 0.0


class TestNdarrayOperations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        he.initialize_fractional()

    def setUp(self):
        self.x = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.y = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.x_enc = he.encrypt(self.x)
        self.y_enc = he.encrypt(self.y)

        self.a = np.random.uniform(low=RANGE_FROM, high=RANGE_TO, size=(ARRAY_SIZE, ARRAY_SIZE))
        self.a_enc = he.encrypt_ndarray(self.a)

    def test_add_ndarray(self):
        r = self.a + self.y
        r_enc = self.a_enc + self.y_enc
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_mul_ndarray(self):
        r = self.a * self.y
        r_enc = self.a_enc * self.y_enc
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_sub_ndarray(self):
        r = self.a - self.y
        r_enc = self.a_enc - self.y_enc
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_neg_ndarray(self):
        r = -self.a
        r_enc = -self.a_enc
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_multiply_ndarray(self):
        r = np.multiply(self.a, self.y)
        r_enc = np.multiply(self.a_enc, self.y_enc)
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_dot_ndarray(self):
        r = np.dot(self.a, self.y)
        r_enc = np.dot(self.a_enc, self.y_enc)
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())


class TestNdarrayOperationsPlain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        he.initialize_fractional()

    def setUp(self):
        self.x = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.y = np.random.uniform(low=RANGE_FROM, high=RANGE_TO)
        self.x_enc = he.encrypt(self.x)
        self.y_enc = he.encrypt(self.y)

        self.a = np.random.uniform(low=RANGE_FROM, high=RANGE_TO, size=(ARRAY_SIZE, ARRAY_SIZE))
        self.a_enc = he.encrypt_ndarray(self.a)

    def test_add_ndarray(self):
        r = self.a + self.y
        r_enc = self.a_enc + self.y
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_mul_ndarray(self):
        r = self.a * self.y
        r_enc = self.a_enc * self.y
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_sub_ndarray(self):
        r = self.a - self.y
        r_enc = self.a_enc - self.y
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_multiply_ndarray(self):
        r = np.multiply(self.a, self.y)
        r_enc = np.multiply(self.a_enc, self.y)
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())

    def test_dot_ndarray(self):
        r = np.dot(self.a, self.y)
        r_enc = np.dot(self.a_enc, self.y)
        r_dec = he.decrypt_ndarray(r_enc)
        r = np.around(r, PRECISION)
        r_dec = np.around(r_dec, PRECISION)
        self.assertSequenceEqual(r.tolist(), r_dec.tolist())


if __name__ == '__main__':
    unittest.main()
