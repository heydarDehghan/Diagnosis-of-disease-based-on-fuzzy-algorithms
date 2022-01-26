from servises import *
from FuzzyClusteringPartTwoAndThree import Equivalence, PartTwoAndThree

if __name__ == "__main__":
    ''' '''
    raw_data = load_data('DataSets/hcvdat0.csv', array=True)
    raw_data[:, 1] = [x[0] for x in raw_data[:, 1]]
    data = Data(data=raw_data, data_range=(4, 13), label_range=1, bias=False, normal=True)

    # model = Equivalence(data)
    #
    # alp = 0.6
    # for x in range(15):
    #     print(f'alpha is --> {alp}')
    #     model.fit(alp)
    #     alp += 0.01
    #     if alp >= 1:
    #         break
    #
    #
    # model.cluster_log.sort(key=lambda x: x.accuracy)
    # best_alpha = model.cluster_log[-1]
    # print(f"best_alpha -----> {best_alpha.alpha_cut}")
    model = PartTwoAndThree(data, 0.61)
    model.fit()

    print("RO")
    print(model.r_ocurrence)
    print("RO\n")

    print("RC")
    print(model.r_confumability)
    print("RC\n")

    pred_train= model.predict(data.trainData)
    acc_tran = accuracy_metric(data.trainLabel, pred_train)
    print(F"Acc train is {acc_tran}")

    pred_test = model.predict(data.testData)
    acc = accuracy_metric(data.testLabel, pred_test)
    print(F"Acc test is {acc}")


