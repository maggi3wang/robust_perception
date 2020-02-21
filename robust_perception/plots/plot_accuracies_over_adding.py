import matplotlib.pyplot as plt
import os
import pandas
import numpy as np

def plot_random():
    # process_num, iter_num, probability_1, probability_2, probability_3, probability_4, probability_5, is_correct, time,
    col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 'is_new_best', 
            'test_class_1', 'test_class_2', 'test_class_3', 'test_class_4', 'test_class_5',
            'blank']

    package_directory = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(package_directory,
        '../data/retrained_with_counterexamples/random1/models/random_00')

    final_training_accs = []
    final_test_accs = []
    final_num_epochs_to_best = []       # don't plot this, just print

    for file in sorted(os.listdir(models_dir)):
        filename = os.path.join(models_dir, file)

        if filename.endswith('.csv'):
            data = pandas.read_csv(filename, delimiter=',', names=col_names, header=0)
            model_epoch = data.epoch.to_numpy().astype(int)
            model_training_loss = data.training_loss.to_numpy().astype(float)
            model_training_acc = data.training_acc.to_numpy().astype(float)
            model_test_acc = data.test_acc.to_numpy().astype(float)
            model_is_new_best = data.is_new_best.to_numpy().astype(int)

            for i, is_new_best in reversed(list(enumerate(model_is_new_best))):
                if is_new_best == 1:
                    # Get new best
                    final_training_accs.append(model_training_acc[i])
                    final_test_accs.append(model_test_acc[i])
                    final_num_epochs_to_best.append(i)
                    break

    print(final_test_accs)
    print(final_num_epochs_to_best)
    print(max(final_test_accs))

    # axes = plt.gca()
    # axes.set_xlim([0, len(epochs)])
    # axes.set_ylim([0.0, 1.0])

    # plt.plot(epochs, training_accs, label='training accs')
    # plt.plot(epochs, test_accs, label='test accs')
    # plt.plot(epochs, counterexample_accs, label='counterexample accs')
    # plt.plot(epochs, training_losses, label='training_loss')

    # plt.legend(loc='lower right')

    # plt.show()

def main():
    plot_random()

if __name__ == "__main__":
    main()