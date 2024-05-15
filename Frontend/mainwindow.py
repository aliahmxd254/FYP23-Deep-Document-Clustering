import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from nbconvert import NotebookExporter, PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import papermill as pm
import subprocess

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("Frontend\mainwindow.ui", self)
        
        self.cluster_graph.setVisible(False)
        self.purity_graph.setVisible(False)
        
        self.showMaximized()
        self.algo.currentIndexChanged.connect(self.update_datasets)
        self.update_datasets()
        self.clusterbtn.clicked.connect(self.run_program)
        
    
    def update_datasets(self):
        selected_algo = self.algo.currentText()
        self.data.clear()
        
        if selected_algo == "Baseline":
            datasets = ["BBC", "Doc50"]
        
        elif selected_algo == "SEDCN":
            datasets = ["ACM", "BBC", "Doc50"]
        
        elif selected_algo == "Transformer":
            datasets = ["ACM", "BBC", "Doc50"]
            
        elif selected_algo == "DTSN":
            datasets = ["ACM", "BBC", "Doc50"]
        
        else:
            datasets = []
        
        self.data.addItems(datasets)
        
    
    def run_program(self):
        self.clusterbtn.setEnabled(False)
        notebook_file = ""
        
        if self.algo.currentText() == "Baseline":
            notebook_file = os.getcwd() + r"\Base_Lines\main_" + str(self.data.currentText()) + ".ipynb"
            with open(notebook_file) as f:
                nb_in = nbformat.read(f, nbformat.NO_CONVERT)
                
            ep = ExecutePreprocessor(timeout=60000, kernel_name='python3')
            nb_out = ep.preprocess(nb_in, resources={'metadata': {'path': os.getcwd()+ r"\Base_Lines"}})
        
        elif self.algo.currentText() == "Transformer":
            if self.data.currentText() == "ACM":
                parameters = {'data_choice': 'acm'}
            elif self.data.currentText() == "BBC":
                parameters = {'data_choice': 'bbc'}
            elif self.data.currentText() == 'Doc50':
                parameters = {'data_choice':'doc50'}
                
            
            notebook_file = os.path.join(os.getcwd(), "Transformer", "Clustering", "TransformerAllDatasets.ipynb")
            # with open(notebook_file) as f:
            #     nb_in = nbformat.read(f, nbformat.NO_CONVERT)
                
            # nb_in.metadata['parameters'] = parameters
            # ep = ExecutePreprocessor(timeout=60000, kernel_name='python3')
            # ep.preprocess(nb_in, {'metadata': {'path': os.path.join(os.getcwd(), "Transformer", "Clustering")}})
            pm.execute_notebook(
            notebook_file,
            None,
            parameters=parameters,
            kernel_name='python3',
            cwd=os.path.join(os.getcwd(), "Transformer", "Clustering")
            ) 
            
            clust = QPixmap(os.path.join(os.getcwd(), "Transformer", "Clustering", "Cluster_image.png"))
            self.cluster_graph.setPixmap(clust)
            
            purity = QPixmap(os.path.join(os.getcwd(), "Transformer", "Clustering", "Purity_image.png"))
            self.purity_graph.setPixmap(purity)
            
            self.cluster_graph.setVisible(True)
            self.purity_graph.setVisible(True)
        
        elif self.algo.currentText() == "SEDCN":
            sedcn_file = os.path.join(os.getcwd(), "sedcn-nn", "sedcn.py")
            if self.data.currentText() == "ACM":
                data_choice = 'acm'
            elif self.data.currentText() == "BBC":
                data_choice = 'bbc'
            elif self.data.currentText() == 'Doc50':
                data_choice = 'doc50'
            
            if os.path.isfile(sedcn_file):
                try:
                    # Run the script using subprocess and pass the parameter
                    result = subprocess.run(['python', sedcn_file, '--name', data_choice], check=True, capture_output=True, text=True)
                    
                    # Print the output and errors (if any)
                    # print(result.stdout)
                    # print(result.stderr)
                    
                except subprocess.CalledProcessError as e:
                    # Handle errors in case the script execution fails
                    print(f"An error occurred while running {sedcn_file}: {e}")
                    print(e.output)
            else:
                print(f"The file {sedcn_file} does not exist.")
        
        elif self.algo.currentText() == "DTSN":
            # D:\FAST\FYP\FYP23-Deep-Document-Clustering\sedcn-nn_Transformer\DTSN.py
            dtsn_file = os.path.join(os.getcwd(), "sedcn-nn_Transformer", "DTSN.py")
            print(dtsn_file)
            if self.data.currentText() == "ACM":
                data_choice = 'acm'
            elif self.data.currentText() == "BBC":
                data_choice = 'bbc'
            elif self.data.currentText() == 'Doc50':
                data_choice = 'doc50'
            
            if os.path.isfile(dtsn_file):
                try:
                    # Run the script using subprocess and pass the parameter
                    result = subprocess.run(['python', dtsn_file, '--name', data_choice], check=True, capture_output=True, text=True)
                    
                    # Print the output and errors (if any)
                    # print(result.stdout)
                    # print(result.stderr)
                    
                except subprocess.CalledProcessError as e:
                    # Handle errors in case the script execution fails
                    print(f"An error occurred while running {dtsn_file}: {e}")
                    print(e.output)
            else:
                print(f"The file {dtsn_file} does not exist.")
        
        with open("Results.txt", 'r') as result_file:
            results = result_file.read()
            self.res.setText(results)
        
        self.clusterbtn.setEnabled(True)
        print("finished running program")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = GUI()
    mainWindow.show()
    sys.exit(app.exec_())
