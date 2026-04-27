import numpy as np

def load_vec(fold_name, file_name):
	"""
	Load the timeseries data from file f'{fold_name}{file_name}.npy' as a 1D numpy array.
	:param fold_name: str, folder name
	:param file_name: str, file name
	:param transpose: whether to transpose the data (AN ARTIFACT OF DATA PREPROCESSING)
	:return: 1D numpy array of shape (n_ts × n_sensors × n_axes_per_sensor, ).
	"""
	return np.load(f'{fold_name}{file_name}.npy').flatten()


def load_mat(fold_name, file_name):
	"""
	Load the timeseries data from file f'{fold_name}{file_name}.npy' as a 2D numpy matrix.
	:param fold_name: str, folder name
	:param file_name: str, file name
	:return: 2D numpy matrix of shape (n_total_measurements, n_ts).
	"""
	return np.load(f'{fold_name}{file_name}.npy')


def get_dataset_mat(fold_name, file_names):
	"""
	Return the data in [f'{fold_name}{file_name}.npy' for file_name in file_names] as a 2D numpy matrix.
	:param fold_name: str, folder name
	:param file_names: [str], list of file names
	:return: 2D numpy matrix of shape (len(file_names), n_ts × n_sensors × n_axes_per_sensor)
	"""
	return np.vstack([load_vec(fold_name, file_name) for file_name in file_names])


def get_dataset_tensor(fold_name, file_names):
	"""
	Return the data in [f'{fold_name}{file_name}.npy' for file_name in file_names] as a 3D numpy matrix.
	:param fold_name: str, folder name
	:param file_names: [str], list of file names
	:return: 3D numpy matrix of shape (len(file_names), n_sensors × n_axes_per_sensor, n_ts)
	"""
	return np.stack([load_mat(fold_name, file_name) for file_name in file_names])


def get_train_test_set(train_fold_name, train_file_names,
                       test_fold_name, test_file_names,
                       as_matrix=True):
	if as_matrix:
		return (get_dataset_mat(train_fold_name, train_file_names),
		        get_dataset_mat(test_fold_name, test_file_names))
	else:
		return (get_dataset_tensor(train_fold_name, train_file_names),
		        get_dataset_tensor(test_fold_name, test_file_names))