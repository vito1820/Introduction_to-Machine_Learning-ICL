Part 1

** run:

Using ‘part1_nn_lib.py’ to run the code.

** note:

‘import matplotlib.pyplot as plt’to visualize the train loss has been commented, and the relevant code can be found in train.
There are three activation functions available  'relu', 'sigmoid' and 'identity'.
And two types of loss functions 'cross_entropy' and 'mse'.
the lr will decay every 500 epochs.

** output:

The epoch and loss every 100 epochs will be printed out on the screen, 
and the final 'Train loss', 'Validation loss', and 'Validation accuracy' as well.
If the code in train is not commented, the train_loss visualization will be saved in root.



Part 2

** run:

Using 'python3 part2_house_value_regression.py' to run the code.


** note:

We are using GPU to do hyper-parameters search, therefore the best model from the first run cannot be excuted on LabTS.
Use the evaluated parameters to run the code on CPU again to generate the final model.

If you want to run the code on CPU, un-comment line 199 and 297 to enable "device = torch.device("cpu")".
Actually we did this to generate model that can be run on CPU.


** output:

Hyper-parameters will be printed out on the screen,
and all the scores under each search will be dumpted into "all.txt" file,
in the order of ['lr', 'max_epoch', 'decay', 'train_mse', 'vali_mse', 'train_rmse', 'vali_rmse', 'train_r2', 'vali_r2'].

Use both to evaluate the best parameters.

The model, "part2_model.pickle", is used to load and predict on other data set.
