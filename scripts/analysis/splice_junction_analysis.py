from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import csv



# read datasets
data_label  = []
data_name   = []
data_seq    = []

with open("../../data/processed/splice_junction_data.tsv", mode="r") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for row in tsv_file:
        data_label.append(row[0])
        data_name.append(row[1])
        data_seq.append(row[2])


class EncodeClass():
    def __init__(self, data):
        self.data = data
    
    @staticmethod
    def onehot_encode(seq):
        bases = ["A", "C", "G", "T"]
        matrix = np.zeros((4, 60))
        for i, base in enumerate(seq):
            if base in bases:
                matrix[bases.index(base), i] += 1
            elif base == "D":
                matrix[:, i] += [1, 0, 1, 1]    # D: A or G or T
            elif base == "N":
                matrix[:, i] += [1, 1, 1, 1]    # N: A or G or C or T
            elif base == "S":
                matrix[:, i] += [0, 1, 1, 0]    # S: C or G
            elif base == "R":
                matrix[:, i] += [1, 0, 1, 0]    # R: A or G
        return matrix
    
    @staticmethod
    def return_encoded(data):
        encoded_sequences = [EncodeClass.onehot_encode(seq) for seq in data]
        return encoded_sequences   # 3190 x 240 の2次元リストで返す
    
    @staticmethod
    def label2vec(label):
        mapping = { "EI" : 0,
                    "IE" : 1,
                    "N"  : 2}
        return [mapping[l] for l in label]

class MachineLearning():
    def __init__(self, label, seq):
        self.label = label
        self.seq = seq
        
    def split_data(label, seq):
        x_train, x_test, y_train, y_test = train_test_split(seq, label, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test
    
    def result_output(y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    @staticmethod
    # one-vs-rest algorithm for 'multi class'  
    def logistic_regression(label, seq):
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000) # max_iterは収束するまでの最大イテレーション
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        result = MachineLearning.result_output(y_pred, y_test)
        return result
    
    @staticmethod
    # random forest
    def random_forest(label, seq):
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accu = accuracy_score(y_test, y_pred)
        return accu
    
    @staticmethod
    # FNN3
    def fnn3(label, seq):
        label = EncodeClass.label2vec(label)
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        sequence_length = len(seq[0])
        model = Sequential([
            Dense(128, activation="relu", input_shape=(sequence_length, )),
            Dense(64, activation="relu"),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=50, batch_size=32)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy
    
    @staticmethod
    def fnn5(label, seq):
        label = EncodeClass.label2vec(label)
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        sequence_length = len(seq[0])
        model = Sequential([
            Dense(128, activation="relu", input_shape=(sequence_length, )),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=50, batch_size=32)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy
    
    @staticmethod
    # CNN3
    def cnn3(label, seq):
        label = EncodeClass.label2vec(label)
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        sequence_length = len(seq[0])
        model = Sequential([
            Conv1D(64, kernel_size=3, activation="relu", input_shape=(sequence_length, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(256, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(3, activation="softmax")
        ])
        # データの形状を適切に変更
        x_train = x_train.reshape(-1, sequence_length, 1)
        x_test = x_test.reshape(-1, sequence_length, 1)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=50, batch_size=32)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy
    
    @staticmethod
    # CNN5
    def cnn5(label, seq):
        label = EncodeClass.label2vec(label)
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        sequence_length = len(seq[0])
        model = Sequential([
            Conv1D(64, kernel_size=3, activation="relu", input_shape=(sequence_length, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(256, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(512, kernel_size=3, activation="relu"),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(3, activation="softmax")
        ])
        # データの形状を適切に変更
        x_train = x_train.reshape(-1, sequence_length, 1)
        x_test = x_test.reshape(-1, sequence_length, 1)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=50, batch_size=32)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy
    
    @staticmethod
    # LSTM (3 layers)
    def lstm(label, seq):
        label = EncodeClass.label2vec(label)
        x_train, x_test, y_train, y_test = MachineLearning.split_data(label, seq)
        sequence_length = len(seq[0])
        model = Sequential()
        model.add(LSTM(50, input_shape=(sequence_length, 1), return_sequences=True))
        model.add(LSTM(40, return_sequences=True))
        model.add(LSTM(30))
        model.add(Dense(3, activation="softmax"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=10, batch_size=32)
        loss, accuracy = model.evaluate(x_test, y_test)
        return accuracy




def main():
    # preparing input data
    encoder = EncodeClass(data_seq)
    onehot = encoder.return_encoded(data_seq)
    onehot = np.array(onehot)
    ppm = np.load("../../data/processed/splice_junction_ppm.npy")
    onehot_ppm = []
    for i in range(len(ppm)):
        onehot_ppm.append(np.concatenate((onehot[i], ppm[i])))
    onehot_ppm = np.array(onehot_ppm)
    onehot = onehot.reshape(onehot.shape[0], -1)
    onehot = onehot.tolist()
    ppm = ppm.reshape(ppm.shape[0], -1)
    ppm = ppm.tolist()
    onehot_ppm = onehot_ppm.reshape(onehot_ppm.shape[0], -1)
    onehot_ppm = onehot_ppm.tolist()
    
    
    # make result
    result = []
    result_index = ["condition", "LR", "RF", "FNN3", "FNN5", "CNN3", "CNN5", "LSTM"]
    
    onehot_lr = MachineLearning.logistic_regression(data_label, onehot)
    onehot_rf = MachineLearning.random_forest(data_label, onehot)
    onehot_fnn3 = MachineLearning.fnn3(data_label, onehot)
    onehot_fnn5 = MachineLearning.fnn5(data_label, onehot)
    onehot_cnn3 = MachineLearning.cnn3(data_label, onehot)
    onehot_cnn5 = MachineLearning.cnn5(data_label, onehot)
    onehot_lstm = MachineLearning.lstm(data_label, onehot)
    onehot_result = ["onehot", onehot_lr, onehot_rf, onehot_fnn3, onehot_fnn5, onehot_cnn3, onehot_cnn5, onehot_lstm]
    
    ppm_lr = MachineLearning.logistic_regression(data_label, ppm)
    ppm_rf = MachineLearning.random_forest(data_label, ppm)
    ppm_fnn3 = MachineLearning.fnn3(data_label, ppm)
    ppm_fnn5 = MachineLearning.fnn5(data_label, ppm)
    ppm_cnn3 = MachineLearning.cnn3(data_label, ppm)
    ppm_cnn5 = MachineLearning.cnn5(data_label, ppm)
    ppm_lstm = MachineLearning.lstm(data_label, ppm)
    ppm_result = ["ppm", ppm_lr, ppm_rf, ppm_fnn3, ppm_fnn5, ppm_cnn3, ppm_cnn5, ppm_lstm]
    
    onehot_ppm_lr = MachineLearning.logistic_regression(data_label, onehot_ppm)
    onehot_ppm_rf = MachineLearning.random_forest(data_label, onehot_ppm)
    onehot_ppm_fnn3 = MachineLearning.fnn3(data_label, onehot_ppm)
    onehot_ppm_fnn5 = MachineLearning.fnn5(data_label, onehot_ppm)
    onehot_ppm_cnn3 = MachineLearning.cnn3(data_label, onehot_ppm)
    onehot_ppm_cnn5 = MachineLearning.cnn5(data_label, onehot_ppm)
    onehot_ppm_lstm = MachineLearning.lstm(data_label, onehot_ppm)
    onehot_ppm_result = ["onehot+ppm", onehot_ppm_lr, onehot_ppm_rf, onehot_ppm_fnn3, onehot_ppm_fnn5, onehot_ppm_cnn3, onehot_ppm_cnn5, onehot_ppm_lstm]
    
    result.append(result_index)
    result.append(onehot_result)
    result.append(ppm_result)
    result.append(onehot_ppm_result)
    
    with open("../../data/output/splice_junction_result.tsv", "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        for row in result:
            writer.writerow(row)
    
    print('"../../data/output/"に結果を出力しました')


if __name__ == "__main__":
    main()
    

