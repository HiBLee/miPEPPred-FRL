import copy
import time
import joblib
import pandas as pd
from EFISS_ES import EFISS_ES_FeatureSelection
from FeatureExtract import *
from MLModel import *
np.set_printoptions(suppress=True)


def modelSave(train_x, train_y, test_x, test_y, column_name):

    clf = svm.SVC(random_state=100, probability=True)
    clf.fit(train_x, train_y)
    joblib.dump(clf, './final_model/miPEPred_FRL_SVM.m')

    pd.DataFrame(np.column_stack((train_y, train_x))).to_csv('./final_model/train_feature.csv', index=False, header=column_name)

    for name in test_x.keys():

        pd.DataFrame(np.column_stack((test_y[name], test_x[name]))).to_csv('./final_model/' + name + '_feature.csv', index=False, header=column_name)

        print('########################### ', name)
        predict_y_class = clf.predict(test_x[name])
        predict_y_pro = clf.predict_proba(test_x[name])[:, 1]
        print("%s: " % name, MetricsCalculate(test_y[name], predict_y_class, predict_y_pro))

if __name__ == '__main__':

    print("start_time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()

    print("training......")

    ath_train_seq = ReadFileFromFasta("./data/train_dataset_500.txt")

    ath_independent_test = ReadFileFromFasta("./data/independent_test_dataset1_607.txt")
    fabaceae_independent_test = ReadFileFromFasta("./data/independent_test_dataset2_612.txt")
    hybirdspecies_independent_test = ReadFileFromFasta("./data/independent_test_dataset3_1018.txt")

    
    train_features, train_labels, train_features_name = FeatureGenerator(ath_train_seq)
    ath_independent_test_features, ath_independent_test_labels, ath_independent_test_features_name = FeatureGenerator(ath_independent_test)
    fabaceae_independent_test_features, fabaceae_independent_test_labels, fabaceae_independent_test_features_name = FeatureGenerator(fabaceae_independent_test)
    hybirdspecies_independent_test_features, hybirdspecies_independent_test_labels, hybirdspecies_independent_test_features_name = FeatureGenerator(hybirdspecies_independent_test)

    test_x = {}
    test_y = {'ath_independent_test': ath_independent_test_labels,
              'fabaceae_independent_test': fabaceae_independent_test_labels,
              'hybirdspecies_independent_test': hybirdspecies_independent_test_labels}

    model_obj_arr = ['DT', 'NB', 'KNN', 'LDA', 'SVM', 'LR',
                     'Bag', 'AB', 'GBDT', 'RF', 'LGBM', 'XGB']

    model_dict = {}
    for key in train_features.keys():
        train_features_bds = train_features[key]
        train_labels_bds = train_labels

        test_x['ath_independent_test'] = ath_independent_test_features[key]
        test_x['fabaceae_independent_test'] = fabaceae_independent_test_features[key]
        test_x['hybirdspecies_independent_test'] = hybirdspecies_independent_test_features[key]

        filtered_feature = EFISS_ES_FeatureSelection(train_features_bds, train_labels_bds, 0.3, 0.5)
        if len(filtered_feature) != 0:
            train_features_bds = train_features_bds[:, filtered_feature]
            test_x['ath_independent_test'] = test_x['ath_independent_test'][:, filtered_feature]
            test_x['fabaceae_independent_test'] = test_x['fabaceae_independent_test'][:, filtered_feature]
            test_x['hybirdspecies_independent_test'] = test_x['hybirdspecies_independent_test'][:, filtered_feature]

            model_ranked = GenerateOptimalBaselineMLModel(train_features_bds, train_labels_bds, test_x, test_y, model_obj_arr, 10, 3, False)
            model_dict[key] = model_ranked

    train_class_feature_aggregate = [[] for j in range(np.shape(train_labels)[0])]
    train_pro_feature_aggregate = [[] for j in range(np.shape(train_labels)[0])]
    test_class_feature_aggregate = {}
    test_pro_feature_aggregate = {}
    for test_dataset in test_x.keys():
        test_class_feature_aggregate[test_dataset] = [[] for j in range(np.shape(test_y[test_dataset])[0])]
        test_pro_feature_aggregate[test_dataset] = [[] for j in range(np.shape(test_y[test_dataset])[0])]

    column_name_c = []
    column_name_p = []
    column_name = []

    train_y_new = []

    for key in model_dict.keys():

        model_name = [(element[0], element[1][5][3]) for element in model_dict[key]]
        print('%s' % key, model_name)
        for model in model_dict[key]:

            train_class_feature = model[1][0]
            train_pro_feature = model[1][1]

            train_class_feature_aggregate = np.column_stack((train_class_feature_aggregate, train_class_feature))
            train_pro_feature_aggregate = np.column_stack((train_pro_feature_aggregate, train_pro_feature))

            for test_dataset in test_x.keys():
                test_class_feature = model[1][2][test_dataset]
                test_class_feature_aggregate[test_dataset] = np.column_stack((test_class_feature_aggregate[test_dataset], test_class_feature))

                test_pro_feature = model[1][3][test_dataset]
                test_pro_feature_aggregate[test_dataset] = np.column_stack((test_pro_feature_aggregate[test_dataset], test_pro_feature))

            column_name_c.append('C_' + key + '_' + model[0])
            column_name_p.append('P_' + key + '_' + model[0] + '_1')

            if np.shape(train_y_new)[0] == 0:
                train_y_new = np.array(model[1][4])
            if np.shape(train_y_new)[0] != 0 and (train_y_new == model[1][4]).all():
                train_y_new = np.array(model[1][4])
            else:
                print("The label sets are not same between different models")
                sys.exit()

    if len(train_y_new) != 0:
        train_y = train_y_new
    else:
        print("No baseline classifier is selected under CSR")
        sys.exit()


    train_class_feature_level1 = train_class_feature_aggregate
    train_pro_feature_level1 = train_pro_feature_aggregate
    test_class_feature_level1 = {'ath_independent_test': test_class_feature_aggregate['ath_independent_test'],
                                 'fabaceae_independent_test': test_class_feature_aggregate['fabaceae_independent_test'],
                                 'hybirdspecies_independent_test': test_class_feature_aggregate['hybirdspecies_independent_test']}
    test_pro_feature_level1 = {'ath_independent_test': test_pro_feature_aggregate['ath_independent_test'],
                               'fabaceae_independent_test': test_pro_feature_aggregate['fabaceae_independent_test'],
                               'hybirdspecies_independent_test': test_pro_feature_aggregate['hybirdspecies_independent_test']}

    column_name.extend(column_name_c)
    column_name.extend(column_name_p)


    train_class_pro_feature_level1 = np.column_stack((train_class_feature_level1, train_pro_feature_level1))
    test_class_pro_feature_level1 = {'ath_independent_test': np.column_stack((test_class_feature_level1['ath_independent_test'], test_pro_feature_level1['ath_independent_test'])),
                                     'fabaceae_independent_test': np.column_stack((test_class_feature_level1['fabaceae_independent_test'], test_pro_feature_level1['fabaceae_independent_test'])),
                                     'hybirdspecies_independent_test': np.column_stack((test_class_feature_level1['hybirdspecies_independent_test'], test_pro_feature_level1['hybirdspecies_independent_test']))}

    column_name.insert(0, 'label')
    pd.DataFrame(np.column_stack((test_y['fabaceae_independent_test'], test_class_pro_feature_level1['fabaceae_independent_test']))).to_csv('./final_model/fabaceae_independent_test_feature.csv', index=False, header=column_name)

    train_class_pro_feature_level1_original = copy.deepcopy(train_class_pro_feature_level1)
    test_class_pro_feature_level1_original = copy.deepcopy(test_class_pro_feature_level1)
    train_y_original = copy.deepcopy(train_y)

    depth_feature, metrics_ranked = CascadeArchitecture(train_class_pro_feature_level1, train_y_original, test_class_pro_feature_level1, test_y, model_obj_arr, 10, 3, 0, column_name)

    if len(depth_feature) > 0:

        arr = depth_feature[metrics_ranked[0][0] - 1]
        train_x = arr[0]
        test_x = arr[1]
        train_y = arr[2]
        column_name = arr[3]
        column_name.insert(0, 'label')
        dep = arr[4]

        print('The depth of cascade architecture: ', dep)

        feature_range = np.shape(train_x)[1] - np.shape(train_class_pro_feature_level1)[1]
        if feature_range == 0:
            feature_range = np.shape(train_x)[1]

        print('The dismension of augmented features: ', feature_range)

        # only using augmented features
        train_x = train_x[:, :feature_range]
        test_x['ath_independent_test'] = test_x['ath_independent_test'][:, :feature_range]
        test_x['fabaceae_independent_test'] = test_x['fabaceae_independent_test'][:, :feature_range]
        test_x['hybirdspecies_independent_test'] = test_x['hybirdspecies_independent_test'][:, :feature_range]

        modelSave(train_x, train_y, test_x, test_y, column_name[:feature_range+1])

    print("end_time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
