import numpy as np

if __name__ == "__main__":

    with open('./result/q1.npy', 'rb') as f:
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print("==== Result ====")
    print("avg_train_history")
    print(avg_train_history)
    print("std_train_history")
    print(std_train_history)
    print()
    print("avg_test_history")
    print(avg_test_history)
    print("std_train_history")
    print(std_test_history)
