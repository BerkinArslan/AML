
import numpy as np
import unittest




def fit(data_path: str) -> np.ndarray:
    if type(data_path) != str:
        raise ValueError("Input must be a string")
    data = np.genfromtxt(data_path, dtype = str, delimiter= '\n')
    data = np.unique(data)
    return data

def encode(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    if type(labels) != np.ndarray:
        raise ValueError("Input must be a numpy array")
    l = labels.shape[0]
    n = np.unique(labels).shape[0]
    if l != n:
        raise ValueError("Input array must contain unique values")
    encoded = np.eye(l)
    encode_labels_dictionary = {index: label for index, label in enumerate(labels)}
    print(f"Encoded labels: {encode_labels_dictionary}")
    print(f"Encoded array: {encoded}")
    return encoded, encode_labels_dictionary


def decode(output_vector: np.ndarray, encode_dictionary: dict) -> np.ndarray:
    if type(output_vector) != np.ndarray:
        raise ValueError("Input must be a numpy array")
    output_vector = output_vector.flatten()
    decoded = np.argmax(output_vector, axis=0)
    if decoded <= 0:
        raise ValueError("Input vector must contain positive values")
    """
    largest = output_vector[0]
    for i in range(len(output_vector)):
        if output_vector[i] > largest:
            largest = output_vector[i]
    decoded = np.zeros(len(output_vector))
    """

    output_element = encode_dictionary[decoded]
    return output_element


#I did not include every possible test, because it takes too much time.
class TestFuctionalOneHotEncoding(unittest.TestCase):
    def test_fit_input(self):
        data = ['a', 1, 'c']
        self.assertRaises(ValueError, fit, data)
    def test_output_is_correct_type(self):
        data = 'bearing_faults.csv'
        self.assertIsInstance(fit(data), np.ndarray)
    def test_unique_values(self):
        data = 'bearing_faults.csv'
        data = fit(data)
        self.assertEqual(data.shape[0], np.unique(data).shape[0])

    def test_encode_input(self):
        data = ['a', 1, 'c']
        self.assertRaises(ValueError, encode, data)
    def test_encode_gets_unique_input(self):
        data = np.array(['a', 'a', 'b'])
        self.assertRaises(ValueError, encode, data)
    def test_encode_output(self):
        data = np.array(['a', 'b', 'c'])
        encoded, encode_labels_dictionary = encode(data)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertIsInstance(encode_labels_dictionary, dict)

    def test_decode_input(self):
        data = ['a', 1, 'c']
        self.assertRaises(ValueError, decode, data, {})
    def test_edge_cases_zero_vector(self):
        data, labels = encode(np.array(['a', 'b', 'c']))
        zeros_vector = np.zeros((3,1))
        self.assertRaises(ValueError, decode, zeros_vector, labels)
    def test_decode_output(self):
        data, labels = encode(np.array(['a', 'b', 'c']))
        output_1 = decode(np.array([0, 1, 0]), labels)
        self.assertEqual(output_1, 'b')
        output_2 = decode(np.array([0.01, 0.9, 0.09]), labels)
        self.assertEqual(output_2, 'b')




