import os

save_dir = "/home/FuDawei/NLP/Machine_Translation/dataset/debug"
## 
def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + "\n".encode('utf8'))

def split_train_dev(file_path, language):
    with open(file_path, "r") as f:
        data = f.read()
        data_list = data.split("\n")
        total = len(data_list)
        train_len = int(total * 0.99)
        train_data, dev_data = data_list[:train_len], data_list[train_len:]

        with open(os.path.join(save_dir, language+"train"), 'wb+') as ff:
            for line in train_data:
                write_to_file(ff, line)
        
        with open(os.path.join(save_dir, language+"dev"), 'wb+') as ffd:
            for line in dev_data:
                write_to_file(ffd, line)

def split_debug(file_path, language):
    with open(file_path, "r") as f:
        data = f.read()
        data_list = data.split("\n")
        total = len(data_list)
        train_len = int(total * 0.01)
        train_data, dev_data = data_list[:train_len], data_list[train_len:]

        with open(os.path.join(save_dir, language+"debug"), 'wb+') as ff:
            for line in train_data:
                write_to_file(ff, line)
        


if __name__ == "__main__":
    # split_train_dev("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.en", "en")
    # split_train_dev("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.fr", "fr")
    split_debug("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.en", "en")
    split_debug("/home/FuDawei/NLP/Machine_Translation/dataset/europarl-v7.fr-en.fr", "fr")




