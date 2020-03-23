import matplotlib.pyplot as plt
import numpy as np
import os
import pandas

class Plots():

    def __init__(self, models_dir):
        self.models_dir = models_dir

        self.col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 
            'counterexample_acc', 'is_new_best', 
            'test_class_1', 'test_class_2', 'test_class_3', 'test_class_4', 'test_class_5', 'counterex_class_3',
            'blank']

        self.col_names = ['epoch', 'training_loss', 'training_acc', 'test_acc', 
            'is_new_best', 'test_class_1', 'test_class_2', 'test_class_3', 'test_class_4', 'test_class_5',
            'blank']

    def model_accuracy_over_added_training_data(self):
        # col_names = ['process_num', 'iter_num', 'probability']
        # TODO figure out why there's a blank

        package_directory = os.path.dirname(os.path.abspath(__file__))
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/cma_es/models')

        final_training_accs = []
        final_test_accs = []
        # final_counterexample_accs = []
        final_num_epochs_to_best = []       # don't plot this, just print
        folder_num = 0

        for folder in sorted(os.listdir(self.models_dir)):
            folder_name = os.path.join(self.models_dir, folder)

            if os.path.isdir(folder_name) and '_' in folder and folder_num < 5:
                final_training_accs.append([])
                final_test_accs.append([])
                final_num_epochs_to_best.append([])

                for file in sorted(os.listdir(folder_name)):
                    if file.endswith('.csv'):
                        filename = os.path.join(folder_name, file)

                        data = pandas.read_csv(filename, delimiter=',', names=self.col_names, header=0)
                        # print('filename', filename)

                        # automate this / clean up
                        model_epoch = data.epoch.to_numpy().astype(int)
                        model_training_loss = data.training_loss.to_numpy().astype(float)
                        model_training_acc = data.training_acc.to_numpy().astype(float)
                        model_test_acc = data.test_acc.to_numpy().astype(float)
                        # model_counterexample_acc = data.counterexample_acc.to_numpy().astype(float)
                        model_is_new_best = data.is_new_best.to_numpy().astype(int)

                        for i, is_new_best in reversed(list(enumerate(model_is_new_best))):
                            if is_new_best == 1:
                                # Get new best
                                # print(final_training_accs)
                                final_training_accs[folder_num].append(model_training_acc[i])
                                final_test_accs[folder_num].append(model_test_acc[i])
                                # final_counterexample_accs.append(model_counterexample_acc[i])
                                final_num_epochs_to_best[folder_num].append(i)
                                break
                folder_num += 1

        # print(final_num_epochs_to_best)
        
        # fig, axes = plt.subplots(nrows=1, ncols=1)
        print(final_training_accs)

        for i in range(folder_num):
            plt.plot(final_training_accs[i], color='r', label='Training accuracies', linewidth=0.5)
            plt.plot(final_test_accs[i], color='g', label='Test accuracies', linewidth=0.5)

        plt.xlim([0, len(final_training_accs[0]) - 1])
        plt.ylim([0.97, 1])

        # axes[1].plot(final_counterexample_accs, color='b', label='Counterexample accuracies')

        # axes[1].set_xlim([0, len(final_training_accs) - 1])
        # axes[1].set_ylim([0.0, 1.0])
        # axes[1].set_xlabel('Number of Counterexamples Added')

        # for axis in axes:
        plt.ylabel('Accuracy')

        plt.xticks(np.arange(0, len(final_training_accs[0]), step=1))

        #plt.grid()
        # axes.legend()
        # axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        #plt.savefig(os.path.join(package_directory, 'model_accuracies_over_counterexamples.png'))
        plt.show()

    def model_accuracy_over_epochs(self):
        """
        Shows accuracies within one csv file.
        """

        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/cma_es/models')
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/initial_training/models')
        # models_dir = os.path.join(package_directory, '../data/retrained_with_counterexamples/random1/models/old/random_00')

        epochs = []
        training_accs = []
        test_accs = []
        # counterexample_accs = []
        training_losses = []

        for file in sorted(os.listdir(self.models_dir)):
            filename = os.path.join(self.models_dir, file)

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
    package_directory = os.path.dirname(os.path.abspath(__file__))
    plot = Plots(models_dir=os.path.join(package_directory, '../data/retrained_with_counterexamples/random1/models'))
    # plot.model_accuracy_over_epochs()
    plot.model_accuracy_over_added_training_data()

if __name__ == "__main__":
    main()