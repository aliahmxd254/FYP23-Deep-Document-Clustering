import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from nbconvert import NotebookExporter, PythonExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("Frontend\mainwindow.ui", self)
        
        self.cluster_graph.setVisible(False)
        self.purity_graph.setVisible(False)
        
        self.showMaximized()
        self.algo.currentIndexChanged.connect(self.update_datasets)
        self.update_datasets()
        self.clusterbtn.clicked.connect(self.run_notebook)
        
    
    def update_datasets(self):
        selected_algo = self.algo.currentText()
        self.data.clear()
        
        if selected_algo == "Baseline":
            datasets = ["BBC", "Doc50", "Reuters"]
        
        elif selected_algo == "SEDCN":
            datasets = ["ACM", "BBC", "Cite", "Reuters"]
        
        elif selected_algo == "Transformer":
            datasets = ["ACM", "BBC", "Cite", "Doc50", "Reuters"]
        
        else:
            datasets = []
        
        
        self.data.addItems(datasets)
        
    
    def run_notebook(self):
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
                parameters = {'data_choice': 1}
            elif self.data.currentText() == "BBC":
                parameters = {'data_choice': 2}
            elif self.data.currentText() == 'Cite':
                parameters = {'data_choice': 4}
            elif self.data.currentText() == 'Doc50':
                parameters = {'data_choice': 5}
            elif self.data.currentText() == 'Reuters':
                parameters = {'data_choice': 3}
                
            
            notebook_file = os.getcwd() + r"\Transformer\Clustering\TransformerAllDatasets.ipynb"
            with open(notebook_file) as f:
                nb_in = nbformat.read(f, nbformat.NO_CONVERT)
                
            nb_in.metadata['parameters'] = parameters
            ep = ExecutePreprocessor(timeout=60000, kernel_name='python3')
            nb_out = ep.preprocess(nb_in)
            # , resources={'metadata': {'path': os.getcwd()+ r"\Transformer\Clustering"}}
        
        elif self.algo.currentText() == "SEDCN":
            sedcn_file = os.getcwd() + r"\sedcn-nn\sedcn.py"
            
        
        with open("Results.txt", 'r') as result_file:
            results = result_file.read()
            self.res.setText(results)
            
        if self.algo.currentText() == "Transformer":
            clust = QPixmap("D:/FAST/FYP/FYP23-Deep-Document-Clustering/Cluster_Image.png")
            self.cluster_graph.setPixmap(clust)
            
            purity = QPixmap("D:/FAST/FYP/FYP23-Deep-Document-Clustering/Purity_Image.png")
            self.purity_graph.setPixmap(purity)
            
            self.cluster_graph.setVisible(True)
            self.purity_graph.setVisible(True)
            
        
        
        self.clusterbtn.setEnabled(True)
        print("exited")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = GUI()
    mainWindow.show()
    sys.exit(app.exec_())
