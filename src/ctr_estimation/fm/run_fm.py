import xlearn as xl
# https://xlearn-doc-cn.readthedocs.io/en/latest/python_api/index.html

train_sample_file = '/data/snlp/zhangjl/datas/ctr/criteo_sample_50w_train_sample_h498000.csv'
test_sample_file = '/data/snlp/zhangjl/datas/ctr/criteo_sample_50w_train_sample_l2000.csv'

# train_sample_file = '/data/snlp/zhangjl/datas/ctr/xlearn/demo/classification/criteo_ctr/small_train.txt'
# test_sample_file = '/data/snlp/zhangjl/datas/ctr/xlearn/demo/classification/criteo_ctr/small_test.txt'


fm_model = xl.create_fm()
fm_model.setTrain(train_sample_file)
fm_model.setValidate(test_sample_file)
# param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'adagrad', 'k':20, 'epoch':3000,  'metric':'acc'}
param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'k':20, 'epoch':3000,  'metric':'auc'}
fm_model.fit(param, "/data/snlp/zhangjl/projects/torch_rec/src/ctr_estimation/fm/model.out")

fm_model.setTest(test_sample_file)  # Set the path of test dataset
fm_model.setSigmoid()
fm_model.predict("/data/snlp/zhangjl/projects/torch_rec/src/ctr_estimation/fm/model.out", "/data/snlp/zhangjl/projects/torch_rec/src/ctr_estimation/fm/output.txt")

# # y, x = svm_read_problem('/data/snlp/zhangjl/datas/ctr/heart_scale.txt')
# y, x = svm_read_problem(sample_file)

# m = train(y[:-200], x[:-200], '-c 1 -s 0')

# p_label, p_acc, p_val = predict(y[-200:], x[-200:], m, '-b 1')
# print(f'p:{p_label}, acc:{p_acc}, pval:{p_val}')

# y_pred = [item[0] for item in p_val]

# auc = roc_auc_score(y[-200:], y_pred)
# print(f'auc is {auc}')