import os
import json
import random
import librosa


'''
test set action_scenario labels:

calendar_query
calendar_remove
play_audiobook
play_game
general_joke
qa_maths
qa_definition
qa_stock
email_addcontact
email_sendemail
datetime_convert
list_remove
transport_taxi
transport_traffic
iot_hue_lightup
iot_coffee
iot_wemo_off
recommendation_movies
alarm_query
music_likeness
music_dislikeness
takeaway_order
audio_volume_down
audio_volume_other

This means that these intents need to be removed from the training and development sets but kept in the test set. The rest of the intents from the test set need to be removed!!
'''

def create_oov_split(data, test_labels, split):
    data_obj = {}
    for elem in data:
        # convert each line to dict
        elem = json.loads(elem)
        scenario = elem["scenario"].rstrip()
        action = elem["action"].rstrip()
        intent = (scenario + "_" + action).rstrip()
        transcript = elem["sentence"].rstrip()
        if split == "test":
            if intent in test_labels:
                for rec in elem["recordings"]:
                    file_id = rec["file"]
                    audio_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
                    # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "length" from the data_obj)
                    y, sr = librosa.load(audio_path)
                    length = librosa.get_duration(y=y, sr=sr)

                    # add to the data dict
                    data_obj[file_id] = {"data_path": audio_path,
                                        "transcript": transcript,
                                        "scenario": scenario,
                                        "action": action,
                                        "intent": intent,
                                        "length": length}
        else:
            if intent not in test_labels:
                for rec in elem["recordings"]:
                        file_id = rec["file"]
                        audio_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
                        # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "length" from the data_obj)
                        y, sr = librosa.load(audio_path)
                        length = librosa.get_duration(y=y, sr=sr)

                        # add to the data dict
                        data_obj[file_id] = {"data_path": audio_path,
                                            "transcript": transcript,
                                            "scenario": scenario,
                                            "action": action,
                                            "intent": intent,
                                            "length": length}
                        
    return data_obj


def create_non_oov_split(data, oov_data, diff_sce2count, diff_sce2id, test_labels):
    data_obj = {}

    # take half of the elements from each scenario form the difficult data
    for sce in diff_sce2count:
        sce_data = diff_sce2id[sce]
        amount_to_take = round(diff_sce2count[sce] / 2)
        sce_data = sce_data[:amount_to_take]
        for elem in sce_data:
            data_obj[elem] = oov_data[elem]
    
    # the other half of the elements should be with same action as in the test set
    for sce in diff_sce2count.keys():
        sce_amount = round(diff_sce2count[sce] / 2)
        for elem in data:
            elem = json.loads(elem)
            scenario = elem["scenario"]
            action = elem["action"]
            intent = (scenario + "_" + action).rstrip()
            transcript = elem["sentence"]

            if sce_amount == 0:
                break
            if sce == scenario and intent in test_labels:
                for rec in elem["recordings"]:
                    file_id = rec["file"]
                    if file_id not in data_obj.keys():
                        audio_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
                        # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "length" from the data_obj)
                        y, sr = librosa.load(audio_path)
                        length = librosa.get_duration(y=y, sr=sr)

                        if sce_amount > 0:
                            # add to the data dict
                            data_obj[file_id] = {"data_path": audio_path,
                                                "transcript": transcript,
                                                "scenario": scenario,
                                                "action": action,
                                                "intent": intent,
                                                "length": length}
                            sce_amount -= 1
                        else:
                            break

        # if we still need to take more of the same scenario
        if sce_amount > 0:
            for elem in data:
                elem = json.loads(elem)
                scenario = elem["scenario"]
                action = elem["action"]
                #intent = elem["intent"]
                intent = (scenario + "_" + action).rstrip()
                transcript = elem["sentence"]

                if sce_amount == 0:
                    break
                if sce == scenario:
                    for rec in elem["recordings"]:
                        file_id = rec["file"]
                        if file_id not in data_obj.keys():
                            audio_path = os.path.join("../../../slurp/scripts/audio/slurp_real", file_id)
                            # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "length" from the data_obj)
                            y, sr = librosa.load(audio_path)
                            length = librosa.get_duration(y=y, sr=sr)

                            if sce_amount > 0:
                                # add to the data dict
                                data_obj[file_id] = {"data_path": audio_path,
                                                    "transcript": transcript,
                                                    "scenario": scenario,
                                                    "action": action,
                                                    "intent": intent,
                                                    "length": length}
                                sce_amount -= 1
                            else:
                                break

    return data_obj


def get_scenario_count(data, oov_data):
    sce2count = {}

    for elem in data:
        scenario = oov_data[elem]["scenario"]
        if scenario not in sce2count:
            sce2count[scenario] = 1
        else:
            sce2count[scenario] += 1
    
    return sce2count


def scenario_to_id(data, from_split):
    if from_split == True:
        sce2id = {}
        for elem in data:
            scenario = data[elem]["scenario"]
            if scenario not in sce2id:
                sce2id[scenario] = [elem]
            else:
                sce2id[scenario].append(elem)
    else:
        sce2id = {}
        for elem in data:
            json_data = json.loads(elem)
            scenario = json_data["scenario"]
            for rec in json_data["recordings"]:
                if scenario not in sce2id:
                    sce2id[scenario] = [rec["file"]]
                else:
                    sce2id[scenario].append(rec["file"])
    
    return sce2id


def save_oov_data(train_data_obj, devel_data_obj, test_data_obj, save_path):
    # Save the splits as JSON files in a SpeechBrain format
    with open(os.path.join(save_path, "train_oov.json"), "w") as f:
        json.dump(train_data_obj, f, indent=4)
    with open(os.path.join(save_path, "dev_oov.json"), "w") as f:
        json.dump(devel_data_obj, f, indent=4)
    with open(os.path.join(save_path, "test.json"), "w") as f:
        json.dump(test_data_obj, f, indent=4)
    
    # Save the splits as ID, scenario CSV files
    with open(os.path.join(save_path, "train_oov.csv"), "w") as f:
        f.write("id,scenario\n")
        for elem in train_data_obj:
            f.write(elem + "," + train_data_obj[elem]["scenario"] + "\n")
    
    with open(os.path.join(save_path, "dev_oov.csv"), "w") as f:
        f.write("id,scenario\n")
        for elem in devel_data_obj:
            f.write(elem + "," + devel_data_obj[elem]["scenario"] + "\n")

    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.write("id,scenario\n")
        for elem in test_data_obj:
            f.write(elem + "," + test_data_obj[elem]["scenario"] + "\n")


def save_non_oov_data(train_data_obj, devel_data_obj, save_path):
    # Save the splits as JSON files in a SpeechBrain format
    with open(os.path.join(save_path, "train_non_oov.json"), "w") as f:
        json.dump(train_data_obj, f, indent=4)
    with open(os.path.join(save_path, "dev_non_oov.json"), "w") as f:
        json.dump(devel_data_obj, f, indent=4)

    # Save the splits as ID, scenario CSV files
    with open(os.path.join(save_path, "train_non_oov.csv"), "w") as f:
        f.write("id,scenario\n")
        for elem in train_data_obj:
            f.write(elem + "," + train_data_obj[elem]["scenario"] + "\n")

    with open(os.path.join(save_path, "dev_non_oov.csv"), "w") as f:
        f.write("id,scenario\n")
        for elem in devel_data_obj:
            f.write(elem + "," + devel_data_obj[elem]["scenario"] + "\n")


if __name__ == "__main__":
    original_slurp_path = "../../../slurp/dataset/slurp"
    save_path = "../../splits/OOV"

    with open(os.path.join(original_slurp_path, "train.jsonl"), "r") as f:
        train_data = f.readlines()
    with open(os.path.join(original_slurp_path, "devel.jsonl"), "r") as f:
        devel_data = f.readlines()
    with open(os.path.join(original_slurp_path, "test.jsonl"), "r") as f:
        test_data = f.readlines()

    test_labels = ["calendar_query", "calendar_remove", "play_audiobook", "play_game", "general_joke", "qa_maths", 
                "qa_definition", "qa_stock", "email_addcontact", "email_sendemail", "datetime_convert", "list_remove", 
                "transport_taxi", "transport_traffic", "iot_hue_lightup", "iot_coffee", "iot_wemo_off", "recommendation_movies", 
                "alarm_query", "music_likeness", "music_dislikeness", "takeaway_order", "audio_volume_down", "audio_volume_other"]


    # Create the OOV splits
    oov_train_data_obj = create_oov_split(train_data, test_labels, "train")
    oov_devel_data_obj = create_oov_split(devel_data, test_labels, "dev")
    oov_test_data_obj = create_oov_split(test_data, test_labels, "test")
    
    # Save the OOV splits
    save_oov_data(oov_train_data_obj, oov_devel_data_obj, oov_test_data_obj, save_path)


    # Create the non-OOV splits
    oov_sce2count_train = get_scenario_count(oov_train_data_obj, oov_train_data_obj)
    oov_sce2id_train = scenario_to_id(oov_train_data_obj, from_split=True)
    all_sce2id_train = scenario_to_id(train_data, from_split=False)
    non_oov_data_obj_train = create_non_oov_split(train_data, oov_train_data_obj, oov_sce2count_train, oov_sce2id_train, test_labels)

    oov_sce2count_devel = get_scenario_count(oov_devel_data_obj, oov_devel_data_obj)
    oov_sce2id_devel = scenario_to_id(oov_devel_data_obj, from_split=True)
    all_sce2id_devel = scenario_to_id(devel_data, from_split=False)
    non_oov_data_obj_devel = create_non_oov_split(devel_data, oov_devel_data_obj, oov_sce2count_devel, oov_sce2id_devel, test_labels)

    # save the non-OOV data
    save_non_oov_data(non_oov_data_obj_train, non_oov_data_obj_devel, save_path)
