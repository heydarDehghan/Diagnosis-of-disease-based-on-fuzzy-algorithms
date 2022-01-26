from servises import *


class Equivalence:

    def __init__(self, data: Data):
        self.data = data
        self.cluster_log = []
        self.dist_array = np.zeros((self.data.trainData.shape[0], self.data.trainData.shape[0]))
        self.transitive_closure = None

        self.implement_dist_array()
        self.calculate_transitive_closure()

    def calculate_dist(self, sample):
        '''
        :param sample: An example that we want to calculate the distance from other samples
        :return:
        '''
        temp = []
        for instance in self.data.trainData:
            dist = np.linalg.norm(sample - instance)
            temp.append(dist)
        return temp

    def implement_dist_array(self):
        '''
        calculate euclidean distance of each sample with other samples
        and normal this distance in end .
        :return:
        '''
        for index, sample in enumerate(self.data.trainData):
            self.dist_array[index] = self.calculate_dist(sample)

        self.dist_array = self.dist_array / np.max(self.dist_array) - np.min(self.dist_array)
        self.dist_array += np.identity(self.dist_array.shape[1])

        print(self.dist_array)

    def calculate_transitive_closure(self):
        composition_r = []
        for row in self.dist_array:
            RT_row = [np.max(np.minimum(row, column)) for column in self.dist_array]
            composition_r.append(RT_row)

        RT = np.maximum(self.dist_array, np.array(composition_r))
        self.transitive_closure = RT

    def do_alpha_cut(self, alpha_cut):
        co_transitive_closure = self.transitive_closure.copy()

        for index, row in enumerate(co_transitive_closure):
            for i, item in enumerate(row):
                co_transitive_closure[index][i] = item >= alpha_cut and 1 or 0

        return co_transitive_closure

    def fit(self, alpha_cut):
        s_i_collection = []
        co_transitive_closure = self.do_alpha_cut(alpha_cut)

        for index, row in enumerate(co_transitive_closure):
            s_i = [i for i, item in enumerate(row) if item == 1]
            s_i_collection.append((index, s_i))

        tolerance_class_list = []
        for (index, s_i) in s_i_collection:
            tolerance_class = []
            for i in s_i:
                flag = True
                for j in tolerance_class:
                    if co_transitive_closure[i][j] != 1:
                        flag = False
                        break
                if flag:
                    tolerance_class.append(i)
            tolerance_class_list.append((index, "".join(list(map(str, tolerance_class)))))

        tolerance_class_list = np.array(tolerance_class_list)
        tolerance_class_list[:, 1] = np.unique(tolerance_class_list[:, 1], return_inverse=True)[1]

        num = len(np.unique(tolerance_class_list[:, 1]))
        acc = accuracy_metric(self.data.trainLabel, list(map(int, tolerance_class_list[:, 1])))
        print(f'number of  cluster with alpha cut  {alpha_cut} is ----> {num}')
        print(f'acc with alpha cut  {alpha_cut} is ----> {acc}')
        info = Info(alpha_cut, acc, num)
        self.cluster_log.append(info)


class PartTwoAndThree:

    def __init__(self, data: Data, alpha_cut):
        self.alpha_cut = alpha_cut
        self.data = data
        self.r_ocurrence = np.zeros((len(self.data.class_list), self.data.trainData.shape[1]))
        self.r_confumability = np.zeros((len(self.data.class_list), self.data.trainData.shape[1]))
        self.r_one_o = np.zeros((data.trainData.shape[0], len(data.class_list)))
        self.r_two_c = np.zeros((data.trainData.shape[0], len(data.class_list)))
        self.transitive_closure = None
        self.calculate_ocurrence_matrix()
        self.calculate_confumability_matrix()

    def calculate_ocurrence_matrix(self):
        """
        We must first determine the mean of each characteristic for each type of disease,
        and store it in the r_ocurrence matrix .that  shape of r_ocurrence is (number_of_class, number_of_future)
        :return: implement r_ocurrenc
        """
        for index, class_label in enumerate(self.data.class_list):
            row_with_class_label = self.data.trainData[self.data.trainLabel == class_label]
            row_r_ocurrence = [np.mean(column) for column in row_with_class_label.T]
            self.r_ocurrence[index] = row_r_ocurrence


    def do_alpha_cut(self):
        for index, row in enumerate(self.data.trainData):
            for i, item in enumerate(row):
                self.data.trainData[index][i] = item >= self.alpha_cut and 1 or 0

    def calculate_confumability_matrix(self):
        """
        :return: implement r_confumability
        """

        self.do_alpha_cut()
        for index, class_label in enumerate(self.data.class_list):
            rows_with_class_label = self.data.trainData[self.data.trainLabel == class_label]
            row_r_confumability = [np.sum(column) / np.sum(rows_with_class_label) for column in
                                   rows_with_class_label.T]

            self.r_confumability[index] = row_r_confumability

        # print(pd.DataFrame(self.r_confumability))

    def calculate_r_one_o(self):
        for index, patient in enumerate(self.data.trainData):
            row = [np.max(np.minimum(patient, row_r_o)) for row_r_o in self.r_ocurrence]
            self.r_one_o[index] = row

    def calculate_r_two_c(self):
        for index, patient in enumerate(self.data.trainData):
            row = [np.max(np.minimum(patient, row_r_c)) for row_r_c in self.r_confumability]
            self.r_two_c[index] = row

    def fit(self):
        self.calculate_r_one_o()
        self.calculate_r_two_c()

    def calculate_r_one_comp(self, sample):

        return [np.max(np.minimum(sample, row_r_one)) for row_r_one in self.r_ocurrence]

    def calculate_r_two_comp(self, sample):

        return [np.max(np.minimum(sample, row_r_two)) for row_r_two in self.r_confumability]

    def predict(self, data):

        predict = []
        for sample in data:
            sample_comp_r_one = np.argmax(self.calculate_r_one_comp(sample))
            sample_comp_r_two = np.argmax(self.calculate_r_one_comp(sample))


            if sample_comp_r_one == sample_comp_r_two:
                predict.append(sample_comp_r_one)
            else:
                predict.append(-1)

        return predict


