import matplotlib.pyplot as plt
import numpy as np
import os
import pandas

class Plots():

    def __init__(self):
        self.col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 
            'counterexample_acc', 'is_new_best', 
            'test_class_1', 'test_class_2', 'test_class_3', 'test_class_4', 'test_class_5', 'counterex_class_3',
            'blank']

        self.col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 
            'is_new_best', 'test_class_1', 'test_class_2', 'test_class_3', 'test_class_4', 'test_class_5',
            'blank']

    def model_accuracy_over_counterexamples(self):
        # col_names = ['process_num', 'iter_num', 'probability']
        # TODO figure out why there's a blank

        package_directory = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/cma_es/models')

        final_training_accs = []
        final_test_accs = []
        final_counterexample_accs = []
        final_num_epochs_to_best = []       # don't plot this, just print

        for file in sorted(os.listdir(models_dir)):
            filename = os.path.join(models_dir, file)

            if filename.endswith('.csv'):
                data = pandas.read_csv(filename, delimiter=',', names=self.col_names, header=0)
                print('filename', filename)

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
        axes[1].set_ylim([0.0, 1.0])
        axes[1].set_xlabel('Number of Counterexamples Added')

        for axis in axes:
            axis.set_ylabel('Accuracy')

            axis.set_xticks(np.arange(0, len(final_training_accs), step=1))

            axis.grid()
            axis.legend()
            # axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.savefig(os.path.join(package_directory, 'model_accuracies_over_counterexamples.png'))

    def model_accuracy_over_epochs(self):
        package_directory = os.path.dirname(os.path.abspath(__file__))
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/cma_es/models')
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/initial_training/models')
        models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/random1/models')
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/random1/models/old/random_00')

        epochs = []
        training_accs = []
        test_accs = []
        # counterexample_accs = []
        training_losses = []

        for file in sorted(os.listdir(models_dir)):
            filename = os.path.join(models_dir, file)

            if filename.endswith('000.csv'):
                data = pandas.read_csv(filename, delimiter=',', names=self.col_names, header=0)
                epochs = data.epoch.to_numpy().astype(float)
                training_accs = data.training_acc.to_numpy().astype(float)
                test_accs = data.test_acc.to_numpy().astype(float)
                # counterexample_accs = data.counterexample_acc.to_numpy().astype(float)
                training_losses = data.training_loss.to_numpy().astype(float)

        axes = plt.gca()
        axes.set_xlim([0, len(epochs)])
        axes.set_ylim([0.0, 1.0])

        plt.plot(epochs, training_accs, label='training accs')
        plt.plot(epochs, test_accs, label='test accs')
        # plt.plot(epochs, counterexample_accs, label='counterexample accs')
        plt.plot(epochs, training_losses, label='training_loss')

        plt.legend(loc='lower right')

        plt.show()

def main():
    # model_accuracy_over_counterexamples()
    plot = Plots()
    plot.model_accuracy_over_epochs()

if __name__ == "__main__":
    main()