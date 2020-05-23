import os
import fnmatch


class ListModel:
    def __init__(self, modelname, path=os.getcwd()):
        self.model_dirs = os.path.join(path, modelname)
        self.list_file = fnmatch.filter(
            os.listdir(self.model_dirs), modelname + '*.hdf5')

    def printList(self):
        for i, file in enumerate(self.list_file, 1):
            print(str(i) + ". " + file)

    def getList(self):
        return self.list_file

    def getFileName(self, index):
        return self.list_file[index - 1]

    def getFilePath(self, index):
        return os.path.join(self.model_dirs, self.list_file[index - 1])

    def count(self):
        return len(self.list_file)
