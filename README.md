# Readme 
This project contains the code necessary to create a GNN using the TrackML Particle Tracking Challenge from kaggle. The data files are in train_100_events directory. In order to train a GNN is necessary first to run the function run_and_save() in /data to graph format/data_to_graph.py. If a master model is required use create_master_files() function in the same file. These two functions save the data into graph format, the only change required in the code is to input the file path where to save the data.

## Train GNN
In order to train a GNN is necessary to use the GNN_train file located in /GNN/. Similarly, the file path to read the data in graph format is needed, as well as a path to save the trained model. There are 9 trained models in /GNN/ models. In order to build the tracks, the GNN_track file needs to be run with similar conditions, updating the path where the data is stored as well as the desired trained model.

## Machine learning models 
There are 3 other machine learning models: XGB, MLP, and CNN. Each of the respective directories contains a code to read the data, train the model test it, and output accuracy and time in order to compare with the GNN.

## Additional information
Each file contains a Jupiter notebook and a python file both files contain the same code, so is up to the user what file to use, and the output will be the same, in order to install the necessary requirements there is a requirements.txt file with all the necessary libraries. The code follows the structure of PREP_1_.pdf report, any aditional infromation or doubdt refer to the report.
