import os
import json


def check_action_overlap(train, dev, test):
    train_actions = []
    dev_actions = []
    test_actions = []
    for elem in train:
        train_actions.append(train[elem]["action_1"] + "+" + train[elem]["action_2"])
    for elem in dev:
        dev_actions.append(dev[elem]["action_1"] + "+" + dev[elem]["action_2"])
    for elem in test:
        test_actions.append(test[elem]["action_1"] + "+" + test[elem]["action_2"])
    

    train_actions = set(train_actions)
    dev_actions = set(dev_actions)
    test_actions = set(test_actions)

    print("Train and Dev: {}".format(len(train_actions.intersection(dev_actions))))
    print("Train and Test: {}".format(len(train_actions.intersection(test_actions))))
    print("Dev and Test: {}".format(len(dev_actions.intersection(test_actions))))
    print("\n")


def check_scenario_overlap(train, dev, test):
    train_scenarios = []
    dev_scenarios = []
    test_scenarios = []
    for elem in train:
        train_scenarios.append(train[elem]["scenario_1"])
    for elem in dev:
        dev_scenarios.append(dev[elem]["scenario_1"])
    for elem in test:
        test_scenarios.append(test[elem]["scenario_1"])

    train_scenarios = set(train_scenarios)
    dev_scenarios = set(dev_scenarios)
    test_scenarios = set(test_scenarios)

    print("Train and Dev: {}".format(len(train_scenarios.intersection(dev_scenarios))))
    print("Train and Test: {}".format(len(train_scenarios.intersection(test_scenarios))))
    print("Dev and Test: {}".format(len(dev_scenarios.intersection(test_scenarios))))
    print("\n")


def check_intent_overlap(train, dev, test):
    train_intents = []
    dev_intents  = []
    test_intents  = []
    for elem in train:
        train_intents.append(train[elem]["intent_1"] + "+" + train[elem]["intent_2"])
    for elem in dev:
        dev_intents.append(dev[elem]["intent_1"] + "+" + dev[elem]["intent_2"])
    for elem in test:
        test_intents.append(test[elem]["intent_1"] + "+" + test[elem]["intent_2"])
    

    train_intents = set(train_intents)
    dev_intents = set(dev_intents)
    test_intents = set(test_intents)

    print("Train and Dev: {}".format(len(train_intents.intersection(dev_intents))))
    print("Train and Test: {}".format(len(train_intents.intersection(test_intents))))
    print("Dev and Test: {}".format(len(dev_intents.intersection(test_intents))))
    print("\n")



def check_id_overlap(train, dev, test):
    train_ids = []
    dev_ids = []
    test_ids = []
    for elem in train:
        train_ids.append(elem)
    for elem in dev:
        dev_ids.append(elem)
    for elem in test:
        test_ids.append(elem)

    train_ids = set(train_ids)
    dev_ids = set(dev_ids)
    test_ids = set(test_ids)

    print("Train and Dev: {}".format(len(train_ids.intersection(dev_ids))))
    print("Train and Test: {}".format(len(train_ids.intersection(test_ids))))
    print("Dev and Test: {}".format(len(dev_ids.intersection(test_ids))))
    print("\n")


def get_class_distribution(train, dev, test):
    train_scenarios = []
    dev_scenarios = []
    test_scenarios = []
    for elem in train:
        train_scenarios.append(train[elem]["scenario_1"])
    for elem in dev:
        dev_scenarios.append(dev[elem]["scenario_1"])
    for elem in test:
        test_scenarios.append(test[elem]["scenario_1"])

    # get scenario counts
    train_sce2count = {}
    dev_sce2count = {}
    test_sce2count = {}

    for intent in train_scenarios:
        if intent not in train_sce2count:
            train_sce2count[intent] = 1
        else:
            train_sce2count[intent] += 1

    for intent in dev_scenarios:
        if intent not in dev_sce2count:
            dev_sce2count[intent] = 1
        else:
            dev_sce2count[intent] += 1

    for intent in test_scenarios:
        if intent not in test_sce2count:
            test_sce2count[intent] = 1
        else:
            test_sce2count[intent] += 1

    # calculate % of each intent in each set with 2 decimals
    train_sce2percent = {}
    dev_sce2percent = {}
    test_sce2percent = {}

    for scenario in train_sce2count:
        train_sce2percent[scenario] = round(train_sce2count[scenario] / len(train_scenarios) * 100, 2)
    for scenario in dev_sce2count:
        dev_sce2percent[scenario] = round(dev_sce2count[scenario] / len(dev_scenarios) * 100, 2)
    for scenario in test_sce2count:
        test_sce2percent[scenario] = round(test_sce2count[scenario] / len(test_scenarios) * 100, 2)

    # order by value
    train_sce2percent = {k: v for k, v in sorted(train_sce2percent.items(), key=lambda item: item[1], reverse=True)}
    dev_sce2percent = {k: v for k, v in sorted(dev_sce2percent.items(), key=lambda item: item[1], reverse=True)}
    test_sce2percent = {k: v for k, v in sorted(test_sce2percent.items(), key=lambda item: item[1], reverse=True)}

    print("Train distribution: {}".format(train_sce2percent))
    print("Dev distribution: {}".format(dev_sce2percent))
    print("Test distribution: {}".format(test_sce2percent))


if __name__ == "__main__":
    data_path = "../../splits/DA_CG"

    with open(os.path.join(data_path, "train_non_cg.json"), "r") as f:
        train = json.load(f)
    with open(os.path.join(data_path, "dev_non_cg.json"), "r") as f:
        dev = json.load(f)
    with open(os.path.join(data_path, "test.json"), "r") as f:
        test = json.load(f)

    print("Train length: ", len(train))
    print("Dev length: ", len(dev))
    print("Test length: ", len(test))
    print("\n")

    print("Action overlap")
    check_action_overlap(train, dev, test)
    print("Intent overlap")
    check_intent_overlap(train, dev, test)
    print("Scenario overlap")
    check_scenario_overlap(train, dev, test)
    print("ID overlap")
    check_id_overlap(train, dev, test)
    print("Class distribution")
    get_class_distribution(train, dev, test)

