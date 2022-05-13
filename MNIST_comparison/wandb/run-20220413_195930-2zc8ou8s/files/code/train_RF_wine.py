import pandas as pd

from WaveletForest import WaveletsForestRegressor
from time import time
from load_mnist import MNISTEnvironment
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(0)


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
	num_wavelets = 10000
	START, END, STEP = 100, 50000, 1000
	print(f'Predicting WF RF')
	train_paths = rf_model.rf.decision_path(X)[0]
	test_paths = rf_model.rf.decision_path(X_test)[0]

	X_pred, _ = rf_model.predict(X, paths=train_paths)
	train_mae = (X_pred.squeeze() == y).sum() / X_pred.shape[0]
	print(f"TOTAL train_mae:{train_mae}")
	total_num_wavelets = len(rf_model.sorted_norms)

	for num_wavelets in tqdm(range(0, total_num_wavelets, 100)):
		print(f"num_wavelets:{num_wavelets}")
		X_pred, _ = rf_model.predict(X, m=num_wavelets, paths=train_paths)
		train_mae = np.power(X_pred.squeeze() - y, 2).sum()
		print(f"train_mae:{train_mae}")

		test_pred, _ = rf_model.predict(X_test, m=num_wavelets, paths=test_paths)#[0].argmax(axis=1)

		test_mae = np.power(test_pred.squeeze() - y_test, 2).sum()
		print(f"test_mae {num_wavelets} wavelets:{test_mae}")

		result = {'num_wavelets':num_wavelets, 'WF_RF_train_mae': train_mae, 'WF_RF_test_mae': test_mae}
		print("***********************")
		try:
			wandb.log(result)
		except:
			print(result)
		# return

def main():
	try:
		wandb.init(project='MNIST_trial_experiments', entity='ibenshaul', mode="online", tags=["wine_dataset"])
	except:
		print("wandb not initialized right")

	X_train, Y_train, X_test, Y_test = create_dataset()
	rf_model = train_tree_model(X_train, Y_train, mode='regression', trees=1, depth=100)
	predict_WF_RF(rf_model, X_train, Y_train, X_test, Y_test)

def create_dataset():
	red_wine = pd.read_csv("wine_dataset/winequality-red.csv", delimiter=";")
	white_wine = pd.read_csv("wine_dataset/winequality-white.csv", delimiter=";")
	train_red, test_red = train_test_split(red_wine, test_size=0.3, random_state=0)
	train_white, test_white = train_test_split(white_wine, test_size=0.3, random_state=0)
	train_total = pd.concat((train_red, train_white))
	test_total = pd.concat((test_red, test_white))

	label_column = 'quality'
	feature_columns = [k for k in train_total.columns if k!=label_column]

	X_train = train_total[feature_columns].values
	Y_train = train_total[[label_column]].values.squeeze()

	X_test = test_total[feature_columns].values
	Y_test = test_total[[label_column]].values.squeeze()
	return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
	main()