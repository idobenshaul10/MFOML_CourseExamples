
from SparsityProbe.SparsityProbe import WaveletsForestRegressor
from time import time
from load_mnist import MNISTEnvironment
from tqdm import tqdm
import wandb

def train_tree_model(x, y, mode='classification',
					 trees=10, depth=15, features='auto',
					 state=2000, nnormalization='volume') -> WaveletsForestRegressor:
	print('Training tree model...')
	wandb.config.update({'trees': trees, 'depth': depth, 'state': state, 'mode':mode})
	model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth,
									seed=state, norms_normalization=nnormalization)

	model.fit(x, y)
	return model

def predict_normal_RF(rf_model, X, y, X_test, y_test):
	print('Predicting normal RF...')
	X_pred = rf_model.rf.predict(X).argmax(axis=1)
	X_test = rf_model.rf.predict(X_test).argmax(axis=1)
	train_acc, test_acc = ((X_pred == y).sum()) / X_pred.shape[0], ((X_test == y_test).sum()) / X_test.shape[0]
	print(f"train_acc:{train_acc}")
	print(f"test_acc:{test_acc}")
	result = {'Normal_RF_train_acc': train_acc, 'Normal_RF_test_acc': test_acc}
	wandb.log(result)
	print("\n")

def predict_WF_RF(rf_model, X, y, X_test, y_test):
	start = time()
	num_wavelets = 100
	print(f'Predicting WF RF')
	X_pred, _ = rf_model.predict(X, m=num_wavelets)
	X_pred = X_pred.argmax(axis=1)
	print(f"train_acc:{((X_pred == y).sum()) / X_pred.shape[0]}")
	test_paths = rf_model.rf.decision_path(X_test)
	for num_wavelets in tqdm(range(30000, 50000, 1000)):
		test_pred = rf_model.predict(X_test, m=num_wavelets)[0].argmax(axis=1)
		test_acc = ((test_pred == y_test).sum()) / X_test.shape[0]
		print(f"test_acc {num_wavelets} wavelets:{test_acc}")
		print(f"took {time() - start}")
		result = {'num_wavelets':num_wavelets, 'Normal_RF_test_acc': test_acc}
		wandb.log(result)

def main():
	wandb.init(project='MNIST_trial_experiments', entity='ibenshaul', mode="online", tags=[])
	environment = MNISTEnvironment()
	_, train_dataset, test_dataset, _ = environment.load_environment()
	X, y = train_dataset.data.reshape((-1, 784)), train_dataset.targets.numpy()
	X_test, y_test = test_dataset.data.reshape((-1, 784)), test_dataset.targets.numpy()
	rf_model = train_tree_model(X, y)
	predict_normal_RF(rf_model, X, y, X_test, y_test)
	predict_WF_RF(rf_model, X, y, X_test, y_test)

	# import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()