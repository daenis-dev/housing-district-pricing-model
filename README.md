# Getting Started

- Install *virtualenv* if it isn't already installed

  ```
  >> py -m pip install --user -U virtualenv
  ```

- Create a virtual environment

  ```
  >> py -m virtualenv env
  ```

- Use Pip to install the required libraries

  ```
  >> py -m pip install -u jupyter matplotlib numpy pandas scipy scikit-learn
  ```

- Train and test the model

  - To analyze the data while training and testing the model, open a Jupyter notebook and run the commands in the file *analyze_and_train.ipynb*

    ```
    >> jupyter notebook 
    ```
    
  - To simply train and test the model, run the *train_model.py* script

    ```
    >> py -m train_model
    ```

