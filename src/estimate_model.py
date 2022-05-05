from typing import Tuple


class EstimateModel:
    def __init__(self, svm_estimator, shape: Tuple, orientation: int, pixels_per_cell: Tuple,
                 cells_per_block: Tuple):
        self.__svm_estimator = svm_estimator
        self.__shape = shape
        self.__orientation = orientation
        self.__pixels_per_cell = pixels_per_cell
        self.__cells_per_block = cells_per_block

    @property
    def svm_estimator(self):
        return self.__svm_estimator

    @property
    def shape(self):
        return self.__shape

    @property
    def orientation(self):
        return self.__orientation

    @property
    def pixels_per_cell(self):
        return self.__pixels_per_cell

    @property
    def cells_per_block(self):
        return self.__cells_per_block
