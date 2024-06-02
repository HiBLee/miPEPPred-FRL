import joblib
import pandas as pd

from MLModel import MetricsCalculate


def Model_Pred(test_path):

    predictor = joblib.load('./final_model/miPEPred_FRL_SVM.m')

    y_true = []
    y_pred_pro = []
    y_pred_class = []


    for path in test_path:
        # 测试集
        test_pd = pd.read_csv(path)
        test_labels = test_pd['label']
        y_true.append(test_labels)
        test_feature = test_pd.drop('label', axis=1).values
        predict_y_pro = predictor.predict_proba(test_feature)[:, 1]
        y_pred_pro.append(predict_y_pro)
        predict_y_class = predictor.predict(test_feature)
        y_pred_class.append(predict_y_class)

    return y_true, y_pred_pro, y_pred_class

if __name__ == '__main__':

    test_path = ['./final_model/ath_independent_test_feature.csv',
                 './final_model/fabaceae_independent_test_feature.csv',
                 './final_model/hybirdspecies_independent_test_feature.csv']

    y_true, y_pred_pro, y_pred_class = Model_Pred(test_path)

    for i in range(len(y_true)):
        print('SN SP PRE ACC MCC F1 AUROC AUPRC', 'TP FN TN FP')
        print(MetricsCalculate(y_true[i], y_pred_class[i], y_pred_pro[i] ))

