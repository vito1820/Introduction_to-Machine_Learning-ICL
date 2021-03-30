# export PYTHONUSERBASE=/vol/lab/ml/mlenv

import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Regressor():

    def __init__(self, x, nb_epoch = 1000, learning_rate = 1e-4, weight_decay = 0.):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # normalise test based on this
        self.train_x_max = 0.
        self.train_x_min = 0.
        self.unique = None
        self.uni_keys = None
        self.model = None
        self.best_hyperparameters = None

        # use x to build one-hot key table
        self.has_one_hot_key = False
        X, _ = self._preprocessor(x, training = True)

        # for tuning
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size 
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size 
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # preprocess x
        if training == True:
            # update one-hot key table
            if self.has_one_hot_key == False:
                # refine as fill nan, and apply one hot key
                x, unique, uni_keys = refine_data(x, training, self.has_one_hot_key)

                # update
                self.unique = unique
                self.uni_keys = uni_keys
                self.has_one_hot_key = True

                # normalise and store train max & min
                self.train_x_max = x.max(axis = 0)
                self.train_x_min = x.min(axis = 0)
                x = (x - self.train_x_min) / (self.train_x_max - self.train_x_min)
                
                # to torch tensor
                x = torch.from_numpy(x.astype('float64'))

            else:
                # only fill nan
                x, _, _ = refine_data(x, training, self.has_one_hot_key)
                non_change = np.array(x)[:, 0:8]
                ocean_proximity = np.array(x)[:, 8]

                # apply one-hot key
                new_label = np.zeros((ocean_proximity.shape[0], 1, self.unique.shape[0]))
                for i in range(ocean_proximity.shape[0]):
                    for j in range(self.unique.shape[0]):
                        if ocean_proximity[i] == self.unique[j]:
                            # assign if match
                            new_label[i] = self.uni_keys[j]

                new_label = new_label.reshape(new_label.shape[0], new_label.shape[2])

                # form x
                x = np.concatenate((non_change, new_label), axis = 1)

                # update train max & min
                self.train_x_max = x.max(axis = 0)
                self.train_x_min = x.min(axis = 0)
                x = (x - self.train_x_min) / (self.train_x_max - self.train_x_min)
                
                # to torch tensor
                x = torch.from_numpy(x.astype('float64'))
            
            # check y None
            if isinstance(y, pd.DataFrame):
                # use $ number k ($200,000 = $200k)
                y = y / 1000.

                # to torch tensor
                my_array = np.array(y)
                y = torch.tensor(my_array)
            
        else:
            # refine as fill nan
            x, _, _ = refine_data(x, training, True)
            non_change = np.array(x)[:, 0:8]
            ocean_proximity = np.array(x)[:, 8]

            # apply one-hot key
            new_label = np.zeros((ocean_proximity.shape[0], 1, self.unique.shape[0]))
            for i in range(ocean_proximity.shape[0]):
                for j in range(self.unique.shape[0]):
                    if ocean_proximity[i] == self.unique[j]:
                        # assign if match
                        new_label[i] = self.uni_keys[j]

            new_label = new_label.reshape(new_label.shape[0], new_label.shape[2])

            # form x
            x = np.concatenate((non_change, new_label), axis = 1)

            # using train max min to normalize
            x = (x - self.train_x_min) / (self.train_x_max - self.train_x_min)

            # to torch tensor
            x = torch.from_numpy(x.astype('float64'))

        
        #print(x)
        #print(y)
        #print(self.train_x_max, self.train_x_min)

        # Return preprocessed x and y, return None for y if it was None
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        print('\n--------------fit--------------')

        # get train x, y
        x, y = self._preprocessor(x, y = y, training = True)

        dtype = torch.float
        # run gpu, cpu if no gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ##################### use cpu to generate model for test
        #device = torch.device("cpu")
        x = x.to(device)
        y = y.to(device)
        print('using device: ', device)
        
        # make 4 hidden layers
        N       = x.shape[0]
        D_in    = x.shape[1]
        H1      = 1024
        H2      = 256
        H3      = 64
        H4      = 16
        D_out   = 1

        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Tensors for its weight and bias.
        print('\n== initialize model')
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H1),
            torch.nn.ReLU(),
            torch.nn.Linear(H1, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, H3),
            torch.nn.ReLU(),
            torch.nn.Linear(H3, H4),
            torch.nn.ReLU(),
            torch.nn.Linear(H4, D_out),
        ).to(device)

        # loss is mse for regression
        loss_fn = torch.nn.MSELoss(reduction='mean')

        # use Adam as adaptive learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        
        # train
        print('\n---------- epoch =', self.nb_epoch, '----------')
        for epoch in range(self.nb_epoch):
            # Forward pass: compute predicted y by passing x to the model.
            train_y_pred = model(x.float())

            # Compute and print loss.
            train_loss = loss_fn(train_y_pred, y.float())
            if (epoch + 1) % 250 == 0:
                #print predict err here
                print('epoch = ', epoch + 1, 'train loss = ', train_loss.item())

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters by autograd
            train_loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

        # store model
        self.model = model

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # normalize validate x with train x max min
        x, _ = self._preprocessor(x, y = None, training = False)

        # get device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ##################### use cpu to generate model for test
        #device = torch.device("cpu")
        x = x.to(device)
        model = self.model.to(device)

        # predict, back to ndarray and change $ 1k to $ 1000
        y_pred = model(x.float()).cpu().detach().numpy() * 1000

        return y_pred

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        
        # predict
        y_pred = self.predict(x)
        y_actual = np.array(y)

        # get scores
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)

        return mse, rmse, r2

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(regressor, data): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # the method is similar to GridSearchCV

    # retrieve hyperparameters
    learning_rate = regressor.learning_rate
    max_epoches = regressor.nb_epoch
    weight_decay = regressor.weight_decay
    output_label = "median_house_value"

    small_rmse = float('inf')
    best_model = None

    # to store results
    info = np.array(['lr', 'max_epoch', 'decay', 'train_mse', 'vali_mse', 'train_rmse', 'vali_rmse', 'train_r2', 'vali_r2']).reshape(1, 9)
    best_info = None

    # greedy loop for each hyperparameter
    for lr in learning_rate:
        for max_epoch in max_epoches:
            for decay in weight_decay:
                #prepare data set for cross validation
                #kfold = KFold(n_splits = 10, shuffle = False, random_state = None)
                kfold = KFold(n_splits = 10, shuffle = True, random_state = None)

                i = 0
                # get separated sets indices
                for (idx_train, idx_validation) in kfold.split(data):
                    i += 1

                    train = data.loc[idx_train]
                    validation = data.loc[idx_validation]

                    # Spliting input and output
                    x_train = train.loc[:, train.columns != output_label]
                    y_train = train.loc[:, [output_label]]

                    x_validation = validation.loc[:, validation.columns != output_label]
                    y_validation = validation.loc[:, [output_label]]

                    # update hyperparameters
                    regressor.nb_epoch = max_epoch
                    regressor.learning_rate = lr
                    regressor.weight_decay = decay
                    print('################################################################')
                    print(i, '/10Fold - ', 'lr=', lr, ' epoch=', max_epoch, ' decay=', decay)

                    # fit current set
                    regressor.fit(x_train, y_train)

                    # get score
                    train_mse, train_rmse, train_r2 = regressor.score(x_train, y_train)
                    validation_mse, validation_rmse, validation_r2 = regressor.score(x_validation, y_validation)
                    print('\n --Scores--\n')
                    print('train MSE  =', train_mse, '\t\tvalidation MSE  =', validation_mse)
                    print('train RMSE =', train_rmse, '\t\tvalidation RMSE =', validation_rmse)
                    print('train R2   =', train_r2, '\t\tvalidation R2   =', validation_r2)

                    # record info
                    temp = np.array([lr, max_epoch, decay, train_mse, validation_mse, train_rmse, validation_rmse, train_r2, validation_r2]).reshape(1, 9)
                    info = np.concatenate((info, temp), axis = 0)

                    # compare RMSE
                    if validation_rmse < small_rmse:
                        small_rmse = validation_rmse
                        best_model = regressor.model
                        best_info = temp

                    print('\n\n')
    

    # catch model
    regressor.model = best_model
    regressor.best_hyperparameters = best_info
    np.savetxt('all.txt', info[1:, :].astype('float64'), fmt = '%f', delimiter = '  ')
    print('All info saved in all.txt')

    # save
    save_regressor(regressor)

    # Return the chosen hyper parameters
    # order is ['lr', 'max_epoch', 'decay', 'train_mse', 'vali_mse', 'train_rmse', 'vali_rmse', 'train_r2', 'vali_r2']
    return best_info

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



# refine as fill nan, and apply one hot key
def refine_data(x, training, has_one_hot_key):
    # get bedrooms column
    total_bedrooms = np.array(x['total_bedrooms'])
    
    # remove nan
    total_bedrooms = total_bedrooms[~np.isnan(total_bedrooms)]
    
    # get total_bedrooms mean
    total_bedrooms_mean = np.sum(total_bedrooms) / float(len(total_bedrooms))
    #print('total_bedrooms_mean = ', total_bedrooms_mean)
    
    # fill nan
    values = {'total_bedrooms': total_bedrooms_mean}
    data = x.fillna(total_bedrooms_mean)

    if training == True and has_one_hot_key == False:
        # one-hot key ['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN']
        one_hot = OneHotEncoder()

        # pick col
        ocean_proximity = np.array(data.loc[:, ['ocean_proximity']])

        # find uni-keys
        unique = np.unique(ocean_proximity)
        unique = unique.reshape(unique.shape[0], 1)
        uni_keys = one_hot.fit_transform(unique).toarray()
        uni_keys = uni_keys.reshape(uni_keys.shape[0], 1, uni_keys.shape[0])

        # build mat
        new_label = np.zeros((ocean_proximity.shape[0], 1, unique.shape[0]))
        for i in range(ocean_proximity.shape[0]):
            for j in range(unique.shape[0]):
                if ocean_proximity[i] == unique[j]:
                    # assign if match
                    new_label[i] = uni_keys[j]

        new_label = new_label.reshape(new_label.shape[0], new_label.shape[2])
        
        # form x
        non_change = np.array(data)[:, 0:8]
        x = np.concatenate((non_change, new_label), axis = 1)

        return x, unique, uni_keys

    else:
        return np.array(data), None, None


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # use x to build one-hot key
    x = data.loc[:, data.columns != output_label]

    # hyperparameters test
    learning_rate   = [0.001, 0.02, 0.05]
    max_epoches     = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    weight_decay    = [0., 0.01, 0.0001]

    # best
    #learning_rate   = [0.02]
    #max_epoches     = [2000]
    #weight_decay    = [0.01]

    # only focus on 2000
    #learning_rate   = [0.005, 0.01, 0.03, 0.04]
    #max_epoches     = [2000]
    #weight_decay    = [0.01]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    print('========== is cuda available: ' ,torch.cuda.is_available())
    regressor = Regressor(x, nb_epoch = max_epoches, learning_rate = learning_rate, weight_decay = weight_decay)

    # get one-hot key table
    #print('----------------one-hot key-----------------\n')
    #print(regressor.unique)
    #print(regressor.uni_keys)

    # find good hyperparameters
    # order is ['lr', 'max_epoch', 'decay', 'train_mse', 'vali_mse', 'train_rmse', 'vali_rmse', 'train_r2', 'vali_r2']
    best_info = RegressorHyperParameterSearch(regressor, data)
    print('Good RegressorHyperParameter')
    print('lr=', best_info[0][0], ' epoch=', best_info[0][1], ' decay=', best_info[0][2])

if __name__ == "__main__":
    example_main()

