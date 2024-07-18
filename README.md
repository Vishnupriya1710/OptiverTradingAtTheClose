# Optiver Trading At The Close
## Description of the notebook
### Importing Libraries

* Data loadong and description
* Exploratory Data Analysis (EDA)
* Data Preparation
* Reducing the memory usage
* Creating dataset with Data loaders
* Train test split
* Creating sequences for model building
* Model Implementation
    *  LSTM 1
    *  LSTM 2
    *  LSTM 3
* Conclusion

## dataset_description()
Takes a DataFrame (df) as input and returns a new DataFrame 'result' summarizing various attributes of the input dataset.
Takes each attribute from the original dataset and checks if they are 'unique', its 'cardinality', has 'null' values etc.
The function returns another dataset that contains the data description of the original dataset

## Exploratory Data Analysis (EDA)
![image](https://github.com/user-attachments/assets/aaaca904-8298-4469-958b-14548cd7327d)

## Correlation matrix for the attributes
![image](https://github.com/user-attachments/assets/0bca3f6a-cf5d-44f7-b0be-814698552174)
Observations:
* None of the features correlate much target
* The correlation between the 'target' variable and the other variables are very less

### Reducing the memory usage
* The method reduce_mem_usage() attempts to reduce its memory usage by changing the data types of columns to lower memory alternatives when possible.
* The function iterates over each column and checks its data type and the range of values it contains. It then converts the column to a more memory-efficient data type if applicable.
* Reducing the memory usage by truncating some of the data types like int64 -> int16


### Create Training Dataset with tensorflow Dataset Framework
This function windowed_dataset takes a time series data, converts it into a TensorFlow Dataset, and processes it into overlapping windows. Each window contains a sequence of data points for the model to learn from (features) and the next data point as the target (label). The dataset is then batched and prefetched for efficient training.

### Building LSTM model 1 

* The `build_model1` function creates a sequential neural network with an input layer for a 3D input shape representing time lag, number of stocks, and a single LSTM layer with 64 units. The model includes dropout regularization, a dense output layer for predicting stock values, and is compiled with mean absolute error (MAE) loss and the Adam optimizer with a specified learning rate.
* The EarlyStopping callback is configured to monitor the validation loss, halt training when the loss stops decreasing (mode='min'), and restore the model to its best weights based on validation performance (restore_best_weights=True) with a specified patience before stopping (patience). The verbose=True setting prints messages during the early stopping process.


### LSTM model 2
* The build_model2 function constructs a sequential neural network with a more complex architecture, including multiple LSTM layers. It has an input layer for a 3D input shape, three LSTM layers with 64 units each (the first two return sequences), dropout regularization between layers, and a dense output layer for predicting stock values. The model is compiled with mean absolute error (MAE) loss and the Adam optimizer with a specified learning rate.

### LSTM model 3
* The build_model3 function constructs a sequential neural network with a simpler architecture compared to previous models. It includes an input layer for a 3D input shape, a single LSTM layer with 128 units, dropout regularization, and a dense output layer for predicting stock values. The model is compiled with mean absolute error (MAE) loss and the Adam optimizer with a specified learning rate.

### Model	MAE score
* Model 1	- 5.388
* Model 2	- 5.840
* Model 3	- 4.892

## Conclusion
The three models were evaluated based on Mean Absolute Error (MAE) scores, with Model 3 achieving the lowest MAE score of 4.892, indicating superior performance compared to Models 1 and 2. Model 1 follows closely with an MAE score of 5.388, while Model 2 lags slightly behind with a higher MAE score of 5.840. In summary, Model 3 demonstrates the best predictive accuracy among the evaluated models.
