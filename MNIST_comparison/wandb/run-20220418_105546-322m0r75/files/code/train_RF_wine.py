import pandas as pd

from WaveletForest import WaveletsForestRegressor
from random_forest_2 import WaveletsForestRegressor2
from time import time
from load_mnist import MNISTEnvironment
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBRegressor
from sklearn import tree, linear_model, ensemble, preprocessing

np.random.seed(0)


def train_tree_model(x, y, mode='classification',
					 trees=10, depth=15, features='auto',
					 state=2000, nnormalization='volume') -> WaveletsForestRegressor:
	print('Training tree model...')
	wandb.config.update({'trees': trees, 'depth': depth, 'state': state, 'mode': mode})
	model = WaveletsForestRegressor(mode=mode, trees=trees, depth=depth,
									seed=state, norms_normalization=nnormalization)
	# model = WaveletsForestRegressor2(trees=trees, depth=depth, seed=state)

	model.fit(x, y)
	return model


def predict_normal_RF(rf_model, X, y, X_test, y_test):
	print('Predicting normal RF...')
	try:
		scaler = rf_model.min_max_scaler
		X, X_test = scaler.transform(X), scaler.transform(X_test)
		X_pred = rf_model.rf.predict(X)[0].squeeze()
		X_test = rf_model.rf.predict(X_test)[0].squeeze()
	except:
		X_pred = rf_model.predict(X)[0].squeeze()
		X_test = rf_model.predict(X_test)[0].squeeze()

	train_acc, test_acc = np.power(X_pred - y, 2).mean(), np.power(X_test - y_test, 2).mean()
	print(f"train_mae:{train_acc}")
	print(f"test_mae:{test_acc}")

	# result = {'Normal_RF_train_acc': train_acc, 'Normal_RF_test_acc': test_acc}
	# wandb.log(result)
	print("\n")


def predict_WF_RF(rf_model, X, y, X_test, y_test):
	start = time()
	num_wavelets = 10000
	print(f'Predicting WF RF')

	X_pred, train_paths = rf_model.predict(X)  # , paths=train_paths)
	# import pdb; pdb.set_trace()
	train_mae = np.power(X_pred.squeeze() - y, 2).mean()

	print(f"TOTAL train_mae:{train_mae}")

	test_pred, test_paths = rf_model.predict(X_test)  # , paths=test_paths)  # [0].argmax(axis=1)
	test_mae = np.power(test_pred.squeeze() - y_test, 2).mean()
	print(f"test_mae with all wavelets:{test_mae}")
	total_num_wavelets = len(rf_model.norms)

	min_WF_RF_test_mae = 12.
	for num_wavelets in tqdm(range(1, total_num_wavelets, total_num_wavelets // 100)):
		print(f"num_wavelets:{num_wavelets}")
		X_pred, _ = rf_model.predict(X, m=num_wavelets, paths=train_paths)
		train_mae = np.power(X_pred.squeeze() - y, 2).mean()
		print(f"train_mae:{train_mae}")

		test_pred, _ = rf_model.predict(X_test, m=num_wavelets, paths=test_paths)  # [0].argmax(axis=1)

		test_mae = np.power(test_pred.squeeze() - y_test, 2).mean()
		print(f"test_mae {num_wavelets} wavelets:{test_mae}")
		min_WF_RF_test_mae = min(test_mae, min_WF_RF_test_mae)

		result = {'num_wavelets': num_wavelets, 'WF_RF_train_mae': train_mae, 'WF_RF_test_mae': test_mae,
				  'min_WF_RF_test_mae': min_WF_RF_test_mae}
		print("***********************")
		try:
			wandb.log(result)
		except:
			print(result)
	# return


def main(seed, trees):
	try:
		wandb.init(project='MNIST_trial_experiments', entity='ibenshaul', mode="online",
				   tags=["wine_dataset"], reinit=True)
		wandb.config.update({'seed': seed})
	except:
		print("wandb not initialized right")

	X_train, Y_train, X_test, Y_test = create_dataset(seed)
	rf_model = train_tree_model(X_train, Y_train, mode='regression', trees=trees, depth=100)

	# rf_model = XGBRegressor().fit(X_train, Y_train)
	# rf_model = model.fit(X_train, Y_train)
	# rf_model = ensemble.RandomForestRegressor(n_estimators=100, max_depth=15).fit(X_train, Y_train)
	predict_WF_RF(rf_model, X_train, Y_train, X_test, Y_test)
	predict_normal_RF(rf_model, X_train, Y_train, X_test, Y_test)


def create_dataset(seed=1):
	red_wine = pd.read_csv("wine_dataset/winequality-red.csv", delimiter=";")
	white_wine = pd.read_csv("wine_dataset/winequality-white.csv", delimiter=";")
	train_red, test_red = train_test_split(red_wine, test_size=0.15, random_state=seed)
	train_white, test_white = train_test_split(white_wine, test_size=0.15, random_state=seed)
	train_total = pd.concat((train_red, train_white))
	test_total = pd.concat((test_red, test_white))
	# train_total = train_red
	# test_total = test_red

	label_column = 'quality'
	feature_columns = [k for k in train_total.columns if k != label_column]

	X_train = train_total[feature_columns].values
	Y_train = train_total[[label_column]].values.squeeze()

	X_test = test_total[feature_columns].values
	Y_test = test_total[[label_column]].values.squeeze()
	return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
	for i in range(0, 5):
		for trees in [1, 2, 5, 10]:
			main(seed=i, trees=trees)
