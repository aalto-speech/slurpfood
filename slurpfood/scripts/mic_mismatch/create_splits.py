import json
import os
import librosa


def load_data(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def fileid_to_slurpid(slurp_data_train, slurp_data_dev, slurp_data_test):
    f_id_s_id = {}
    for elem in slurp_data_train:
        slurp_id = elem["slurp_id"]
        for rec in elem["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id
    for elem in slurp_data_dev:
        slurp_id = elem["slurp_id"]
        for rec in elem["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id
    for elem in slurp_data_test:
        slurp_id = elem["slurp_id"]
        for rec in elem["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id
    
    return f_id_s_id


def save_splits(save_path, data_obj_train, data_obj_dev, data_obj_headset, data_obj_other, f_id_s_id):
    # Save as Json files in a SpeechBrain JSON format
    with open(os.path.join(save_path, "train.json"), "w") as f:
        json.dump(data_obj_train, f, indent=4)
    with open(os.path.join(save_path, "dev.json"), "w") as f:
        json.dump(data_obj_dev, f, indent=4)
    with open(os.path.join(save_path, "test_headset.json"), "w") as f:
        json.dump(data_obj_headset, f, indent=4)
    with open(os.path.join(save_path, "test_other.json"), "w") as f:
        json.dump(data_obj_other, f, indent=4)

    # Save as ID,label CSV file
    with open(os.path.join(save_path, "train.csv"), "w") as f:
        f.write("slurp_id,file_id,scenario\n")
        for elem in data_obj_train:
            slurp_id = str(f_id_s_id[elem])
            f.write(slurp_id + "," + elem + "," + data_obj_train[elem]["scenario"] + "\n")

    with open(os.path.join(save_path, "dev.csv"), "w") as f:
        f.write("slurp_id,file_id,scenario\n")
        for elem in data_obj_dev:
            slurp_id = str(f_id_s_id[elem])
            f.write(slurp_id + "," + elem + "," + data_obj_dev[elem]["scenario"] + "\n")

    with open(os.path.join(save_path, "test_headset.csv"), "w") as f:
        f.write("slurp_id,file_id,scenario\n")
        for elem in data_obj_headset:
            slurp_id = str(f_id_s_id[elem])
            f.write(slurp_id + "," + elem + "," + data_obj_headset[elem]["scenario"] + "\n")

    with open(os.path.join(save_path, "test_other.csv"), "w") as f:
        f.write("slurp_id,file_id,scenario\n")
        for elem in data_obj_other:
            slurp_id = str(f_id_s_id[elem])
            f.write(slurp_id + "," + elem + "," + data_obj_other[elem]["scenario"] + "\n")


def create_splits(save_path, train_data, dev_data, test_data, f_id_s_id):
    '''
    This function is used to create the training and development and testing splits.
    Train and dev splits contain only the recordings made with a headset.
    Test split is divided in two portions. An easy one that has recordings made with a headset and a hard one that has recordings without a headset.
    '''
    data_obj_train = {}
    data_obj_dev = {}
    data_obj_headset = {}
    data_obj_other = {}

    # Create train split
    for elem in train_data:
        transcript = elem["sentence"].rstrip()
        action = elem["action"]
        scenario = elem["scenario"]
        intent = scenario + "_" + action
        for recording in elem["recordings"]:
            file_id = recording["file"]
            data_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
            # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remove "length" from the data_obj)
            y, sr = librosa.load(data_path)
            length = librosa.get_duration(y=y, sr=sr)

            if "headset" in file_id:
                data_obj_train[file_id] = {"data_path": data_path, 
                                "transcript": transcript, 
                                "scenario": scenario, 
                                "action": action,
                                "intent": intent,
                                "length": length}
    
    # Create dev split
    for elem in dev_data:
        transcript = elem["sentence"].rstrip()
        action = elem["action"]
        scenario = elem["scenario"]
        intent = scenario + "_" + action
        for recording in elem["recordings"]:
            file_id = recording["file"]
            data_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
            # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remove "length" from the data_obj)
            y, sr = librosa.load(data_path)
            length = librosa.get_duration(y=y, sr=sr)

            if "headset" in file_id:
                data_obj_dev[file_id] = {"data_path": data_path, 
                                "transcript": transcript, 
                                "scenario": scenario, 
                                "action": action,
                                "intent": intent,
                                "length": length}

    # Create test splits
    for elem in test_data:
        transcript = elem["sentence"].rstrip()
        action = elem["action"]
        scenario = elem["scenario"]
        intent = scenario + "_" + action
        headset_files = []
        other_files = []
        for recording in elem["recordings"]:
            file_id = recording["file"]

            if "-headset" in file_id:
                headset_files.append(file_id)
            else:
                other_files.append(file_id)
        
        # remove the headset part from the file name
        headset_files = [file.replace("-headset", "") for file in headset_files]

        same_files = []
        for file in headset_files:
            if file in other_files:
                same_files.append(file)
        
        for file in same_files:
            difficult_file_id = file
            data_path_difficult = os.path.join("../../../slurp/scripts/audio/slurp_real", difficult_file_id)
            # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remove "length" from the data_obj)
            y, sr = librosa.load(data_path)
            length = librosa.get_duration(y=y, sr=sr)

            data_obj_other[difficult_file_id] = {"data_path": data_path_difficult, 
                                "transcript": transcript, 
                                "scenario": scenario, 
                                "action": action,
                                "intent": intent,
                                "length": length}

            headset_file_id = file.split(".flac")[0] + "-headset.flac"
            data_path_headset = os.path.join("../../../slurp/scripts/audio/slurp_real", headset_file_id)

            data_obj_headset[headset_file_id] = {"data_path": data_path_headset, 
                                "transcript": transcript, 
                                "scenario": scenario, 
                                "action": action,
                                "intent": intent,
                                "length": length}


    # Save the splits
    save_splits(save_path, data_obj_train, data_obj_dev, data_obj_headset, data_obj_other, f_id_s_id)



if __name__ == "__main__":
    path_to_slurp = "../../../slurp/dataset/slurp" # original slurp data
    save_path = "../../splits/mic_mismatch" # path to save the splits

    train_data = load_data(os.path.join(path_to_slurp, "train.jsonl"))
    dev_data = load_data(os.path.join(path_to_slurp, "devel.jsonl"))
    test_data = load_data(os.path.join(path_to_slurp, "test.jsonl"))

    # Get file_id to slurp_id mapping
    f_id_s_id = fileid_to_slurpid(train_data, dev_data, test_data)

    create_splits(save_path, train_data, dev_data, test_data, f_id_s_id)
