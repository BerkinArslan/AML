import OOP_OHE
import unittest
import numpy as np

class TestOHEOOP(unittest.TestCase):
    def test_fit_input(self):
        data = ['a', 1, 'c']
        encoder = OOP_OHE.OneHotEncoding() #encoder object created
        self.assertRaises(ValueError, encoder.fit, data)
    def test_fit_output(self):
        data = 'bearing_faults.csv'
        encoder = OOP_OHE.OneHotEncoding()
        self.assertIsInstance(encoder.fit(data), np.ndarray)
    def test_fit_unique_values(self):
        data = 'bearing_faults.csv'
        encoder = OOP_OHE.OneHotEncoding()
        data = encoder.fit(data)
        self.assertEqual(data.shape[0], np.unique(data).shape[0])
    def test_fit_empty_data(self):
        data = 'empty_data.csv'
        encoder = OOP_OHE.OneHotEncoding()
        self.assertIsInstance(encoder.fit(data), np.ndarray)

    def test_encode_input(self):
        data = ['a', 1, 'c']
        encoder = OOP_OHE.OneHotEncoding()
        self.assertRaises(ValueError, encoder.encode, data)
    def test_encode_gets_unique_input(self):
        data = ['a', 'a', 'b']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        self.assertRaises(ValueError, encoder.encode, data)
    def test_output_type(self):
        data = ['a', 1, 'c']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        self.assertIsInstance(encoder.encode(data), tuple)
        self.assertIsInstance(encoder.encode(data)[0], np.ndarray)
        self.assertIsInstance(encoder.encode(data)[1], dict)

    def test_decode_input(self):
        data = ['a', 'b', 'c']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        one_hot, label_dict = encoder.encode(data)
        output_vector = [0, 0, 1]
        self.assertRaises(ValueError, encoder.decode, output_vector, label_dict)
    def test_decode_edge_case_all_null(self):
        data = ['a', 'b', 'c']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        one_hot, label_dict = encoder.encode(data)
        output_vector = [0, 0, 0]
        output_vector = np.array(output_vector)
        self.assertRaises(ValueError, encoder.decode, output_vector, label_dict)
    def test_assert_all_values_equal_one(self):
        data = ['a', 'b', 'c']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        one_hot, label_dict = encoder.encode(data)
        output_vector = [0, 0, 0]
        output_vector = np.array(output_vector)
        self.assertRaises(ValueError, encoder.decode, output_vector, label_dict)
        output_vector = [0.5, 0.5, 0.5]
        output_vector = np.array(output_vector)
        self.assertRaises(ValueError, encoder.decode, output_vector, label_dict)
    def test_correct_output(self):
        data = ['a', 'b', 'c']
        data = np.array(data)
        encoder = OOP_OHE.OneHotEncoding()
        one_hot, label_dict = encoder.encode(data)
        output_vector = [0, 1, 0]
        output_vector = np.array(output_vector)
        self.assertEqual(encoder.decode(output_vector, label_dict),  'b')
        output_vector = [0.29, 0.26, 0.45]
        output_vector = np.array(output_vector)
        self.assertEqual(encoder.decode(output_vector, label_dict), 'c')



