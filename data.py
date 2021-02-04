from torch.utils.data import Dataset
import csv

class ClassifierDataset(Dataset):
    def __init__(self, path):
        self.x, self.y = self.open_file(path)
        self.size = len(self.x)

    def __getitem__(self, index: int):
        if self.y is not None:
            return (self.x[index], self.y[index])
        else:
            return self.x[index]
    def __len__(self) -> int:
        return len(self.x)

    @staticmethod
    def open_file(path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            text = []
            label = []
            next(reader, None)
            for data in reader:
                if len(data) == 2:
                    text.append(data[1])
                    label.append(data[0])
                else:
                    text.append(data[0])
                    label = None
            if label is not None:
                assert len(text) == len(label)
        return text, label