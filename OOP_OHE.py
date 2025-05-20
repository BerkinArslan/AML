import numpy as np

class OneHotEncoding():

    #init olusturabilirsin buaraya
    #classlari olusturan attributelar da datanin attributelari bu sayede o spesifik obje
    #icin bu variablelar kaydedilmis olacak
    #bin tane variable ile ugrasmak zorunda kalmayacaksin her pbje icin özel vari seciceksin
    #ise yarar ama aslinda cok daönemli degil
    #bu sekilde yaparsan eger bu OOPnin avantajini kullanmis olursun
    def fit(self, data_path: str) -> np.ndarray:
        if type(data_path) != str:
            raise ValueError('Input path must be in string')
        data = np.genfromtxt(data_path, dtype = str, delimiter = '\n')
        data = np.unique(data)
        print(data)
        return data

    def encode(self, data: np.ndarray) -> tuple[np.ndarray, dict]:
        if type(data) != np.ndarray:
            raise ValueError('Input data must be numpy array')
        if data.shape[0] != np.unique(data).shape[0]:
            raise ValueError('Input array must contain unique labels')
        n = data.shape[0]
        one_hot_matrix = np.eye(n)
        index_to_label = {index: label for index, label in enumerate(data)}
        return one_hot_matrix, index_to_label


    def decode(self, one_hot_vector: np.ndarray, labels_dictionary: dict) -> str:
        if type(one_hot_vector) != np.ndarray:
            raise ValueError('OneHot Vector should be a numpy array!')
        if max(one_hot_vector) == 0 and min(one_hot_vector) == 0:
            raise ValueError('Input vector must not be all zero')
        total = one_hot_vector.sum()
        if ((1 - total) ** 2) ** 0.5 > 0.01:
            raise ValueError('Sum of entries in input vector must be =1')
        index_of_max = 0
        l = one_hot_vector.shape[0]
        for i in range(l):
            if one_hot_vector[i] > one_hot_vector[index_of_max]:
                index_of_max = i
        return labels_dictionary[index_of_max]