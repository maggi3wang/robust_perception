import matplotlib.pyplot as plt
import numpy as np
import os
import pandas

def model_accuracy_over_counterexamples():
    # col_names = ['process_num', 'iter_num', 'probability']
    # TODO figure out why there's a blank
    col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 
        'counterexample_acc', 'is_new_best', 'blank']

    package_directory = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(package_directory, '../data/experiment1/models')

    final_training_accs = []
    final_test_accs = []
    final_counterexample_accs = []
    final_num_epochs_to_best = []       # don't plot this, just print

    for file in sorted(os.listdir(models_dir)):
        filename = os.path.join(models_dir, file)

        if filename.endswith('.csv'):
            data = pandas.read_csv(filename, delimiter=',', names=col_names, header=1)

            # automate this / clean up
            model_epoch = data.epoch.to_numpy().astype(int)
            model_training_loss = data.training_loss.to_numpy().astype(float)
            model_training_acc = data.training_acc.to_numpy().astype(float)
            model_test_acc = data.test_acc.to_numpy().astype(float)
            model_counterexample_acc = data.counterexample_acc.to_numpy().astype(float)
            model_is_new_best = data.is_new_best.to_numpy().astype(int)

            for i, is_new_best in reversed(list(enumerate(model_is_new_best))):
                if is_new_best == 1:
                    # Get new best
                    final_training_accs.append(model_training_acc[i])
                    final_test_accs.append(model_test_acc[i])
                    final_counterexample_accs.append(model_counterexample_acc[i])
                    final_num_epochs_to_best.append(i)
                    break

    print(final_num_epochs_to_best)
    
    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].plot(final_training_accs, color='r', label='Training accuracies')
    axes[0].plot(final_test_accs, color='g', label='Test accuracies')

    axes[0].set_xlim([0, len(final_training_accs) - 1])
    axes[0].set_ylim([0.98, 1])

    axes[1].plot(final_counterexample_accs, color='b', label='Counterexample accuracies')

    axes[1].set_xlim([0, len(final_training_accs) - 1])
    axes[1].set_ylim([0.0, 0.05])
    axes[1].set_xlabel('Number of Counterexamples Added')

    for axis in axes:
        axis.set_ylabel('Accuracy')

        axis.set_xticks(np.arange(0, len(final_training_accs), step=1))

        axis.grid()
        axis.legend()
        # axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(os.path.join(package_directory, 'model_accuracies_over_counterexamples.png'))

def main():
    model_accuracy_over_counterexamples()

if __name__ == "__main__":
    main()