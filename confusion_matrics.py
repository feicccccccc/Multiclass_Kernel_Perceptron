import numpy as np
import matplotlib.pyplot as plt

class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def plot_confusion_matrix(all_matrix, class_names=class_name):
    """
    Plot the confusion Matrics
    # Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    :param matrix: Confusion Matrics, can be computed from sklearn
    :param class_names: Class name
    :return: Figure object from plt
    """
    # This is really bad not having the diagonal element. I am not comfortable with it.
    matrix = np.mean(all_matrix, axis=2)
    std_matrix = np.std(all_matrix, axis=2)

    np.fill_diagonal(matrix, 0)
    np.fill_diagonal(std_matrix, 0)

    figure = plt.figure(figsize=(15, 15))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = matrix.max() / 2.

    # Plot
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            result_text = "{:.2f} ± {:.2f}".format(matrix[i, j], std_matrix[i, j])
            plt.text(j, i, result_text, horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure
