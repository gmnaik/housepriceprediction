import papermill as pm

# Function to run the Jupyter notebook
def run_notebook():
    input_notebook = 'notebook/House_Price_Prediction_Imputation_EDA.ipynb'  # Update path if necessary
    
    output_notebook = 'notebook/House_Price_Prediction_Imputation_EDA.ipynb'  # Optional, for debugging or logs
    
    # Execute the notebook
    pm.execute_notebook(
        input_notebook,           # Input notebook file path
        output_notebook           # Output notebook file path (for debugging)
    )
    
    print("Notebook executed successfully!")

if __name__ == '__main__':
    run_notebook()
