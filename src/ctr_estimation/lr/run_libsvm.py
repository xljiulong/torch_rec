from libsvm.svmutil import *
from liblinear.liblinearutil import svm_read_problem
from sklearn.metrics import roc_auc_score

sample_file = '/data/snlp/zhangjl/datas/ctr/criteo_sample_50w_train_sample.csv'
# y, x = svm_read_problem('/data/snlp/zhangjl/datas/ctr/heart_scale.txt')
y, x = svm_read_problem(sample_file)

m = svm_train(y[:-200], x[:-200], '-c 1 -s 0')

p_label, p_acc, p_val = svm_predict(y[-200:], x[-200:], m, '-b 1')
print(f'p:{p_label}, acc:{p_acc}, pval:{p_val}')
 
y_pred = [item[0] for item in p_val]

auc = roc_auc_score(y[-200:], y_pred)
print(f'auc is {auc}')