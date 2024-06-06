import json
import os


def get_class_distribution(oov_data, non_oov_data):
    oov_scenarios = []
    non_oov_data_scenarios = []
    for elem in oov_data:
        oov_scenarios.append(oov_data[elem]["scenario"])
    for elem in non_oov_data:
        non_oov_data_scenarios.append(non_oov_data[elem]["scenario"])

    # get scenario counts
    oov_sce2count = {}
    non_oov_sce2count = {}

    for scenario in oov_scenarios:
        if scenario not in oov_sce2count:
            oov_sce2count[scenario] = 1
        else:
            oov_sce2count[scenario] += 1
    
    for scenario in non_oov_data_scenarios:
        if scenario not in non_oov_sce2count:
            non_oov_sce2count[scenario] = 1
        else:
            non_oov_sce2count[scenario] += 1

    # calculate % of each intent in each set with 2 decimals
    oov_sce2percent = {}
    non_oov_sce2percent = {}

    for scenario in oov_scenarios:
        oov_sce2percent[scenario] = round((oov_sce2count[scenario] / len(oov_data)) * 100, 2)
    for scenario in non_oov_data_scenarios:
        non_oov_sce2percent[scenario] = round((non_oov_sce2count[scenario] / len(non_oov_data)) * 100, 2)


    # order by value
    oov_sce2percent = dict(sorted(oov_sce2percent.items(), key=lambda item: item[1], reverse=True))
    non_oov_sce2percent = dict(sorted(non_oov_sce2percent.items(), key=lambda item: item[1], reverse=True))

    print("OOV split distribution: {}".format(oov_sce2percent))
    print("Non-OOV split distribution: {}".format(non_oov_sce2percent))


if __name__ == "__main__":
    data_path = "../../splits/OOV"

    with open(os.path.join(data_path, "train_oov.json"), "r") as f:
        train_data_oov = json.load(f)
    with open(os.path.join(data_path, "dev_oov.json"), "r") as f:
        dev_data_oov = json.load(f)
    with open(os.path.join(data_path, "test.json"), "r") as f:
        test_data = json.load(f)
    
    with open(os.path.join(data_path, "train_non_oov.json"), "r") as f:
        train_data_non_oov = json.load(f)
    with open(os.path.join(data_path, "dev_non_oov.json"), "r") as f:
        dev_data_non_oov = json.load(f)


    # Get class distribution
    get_class_distribution(train_data_oov, train_data_non_oov)
    get_class_distribution(dev_data_oov, dev_data_non_oov)
