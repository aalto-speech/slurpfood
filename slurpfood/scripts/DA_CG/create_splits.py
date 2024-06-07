import os
import json
import random
from itertools import product, permutations
import librosa

# set a random seed
random.seed(2024)

test_intents = [
    ("calendar_set", "calendar_remove"), 
    ("play_music", "play_podcasts"),
    ("play_music", "play_game"),
    ("play_radio", "play_audiobook"),
    ("general_quirky", "general_greet"),
    ("qa_factoid", "qa_currency"),
    ("qa_factoid", "qa_maths"),
    ("qa_definition", "qa_currency"),
    ("email_query", "email_querycontact"),
    ("email_sendemail", "email_addcontact"),
    ("transport_query", "transport_traffic"),
    ("transport_ticket", "transport_taxi"),
    ("lists_query", "lists_createoradd"),
    ("recommendation_events", "recommendation_movies"),
    ("alarm_set", "alarm_remove"),
    ("music_query", "music_settings"),
    ("music_likeness", "music_dislikeness"),
    ("iot_hue_lightoff", "iot_coffee"),
    ("iot_hue_lightoff", "iot_hue_lightup"),
    ("iot_hue_lightoff", "iot_wemo_off"),
    ("iot_hue_lightoff", "iot_hue_lighton"),
    ("iot_hue_lightchange", "iot_cleaning"),
    ("iot_hue_lightchange", "iot_hue_lightdim"),
    ("iot_hue_lightchange", "iot_wemo_on"),
    ("iot_coffee", "iot_cleaning"),
    ("iot_coffee", "iot_wemo_off"),
    ("iot_coffee", "iot_hue_lighton"),
    ("iot_cleaning", "iot_hue_lightdim"),
    ("iot_cleaning", "iot_wemo_on"),
    ("iot_hue_lightup", "iot_wemo_off"),
    ("iot_hue_lightup", "iot_hue_lighton"),
    ("iot_hue_lightdim", "iot_wemo_on"),
    ("iot_wemo_off", "iot_wemo_on"),
    ("audio_volume_mute", "audio_volume_down"),
    ("audio_volume_up", "audio_volume_other")
]

train_intents = [
    ("calendar_set", "calendar_query"), 
    ("play_music", "play_radio"),
    ("play_music", "play_audiobook"),
    ("play_radio", "play_podcasts"),
    ("play_radio", "play_game"),
    ("general_quirky", "general_joke"),
    ("qa_factoid", "qa_definition"),
    ("qa_factoid", "qa_stock"),
    ("qa_definition", "qa_stock"),
    ("qa_definition", "qa_maths"),
    ("email_query", "email_sendemail"),
    ("email_query", "email_addcontact"),
    ("email_sendemail", "email_querycontact"),
    ("transport_query", "transport_ticket"),
    ("transport_query", "transport_taxi"),
    ("transport_ticket", "transport_traffic"),
    ("lists_query", "lists_remove"),
    ("recommendation_events", "recommendation_locations"),
    ("alarm_set", "alarm_query"),
    ("music_query", "music_likeness"),
    ("music_query", "music_dislikeness"),
    ("music_likeness", "music_settings"),
    ("iot_hue_lightoff", "iot_hue_lightchange"),
    ("iot_hue_lightoff", "iot_cleaning"),
    ("iot_hue_lightoff", "iot_hue_lightdim"),
    ("iot_hue_lightoff", "iot_wemo_on"),
    ("iot_hue_lightchange", "iot_coffee"),
    ("iot_hue_lightchange", "iot_hue_lightup"),
    ("iot_hue_lightchange", "iot_wemo_off"),
    ("iot_hue_lightchange", "iot_hue_lighton"),
    ("iot_coffee", "iot_hue_lightup"),
    ("iot_coffee", "iot_hue_lightdim"),
    ("iot_coffee", "iot_wemo_on"),
    ("iot_cleaning", "iot_hue_lightup"),
    ("iot_cleaning", "iot_wemo_off"),
    ("iot_cleaning", "iot_hue_lighton"),
    ("iot_hue_lightup", "iot_hue_lightdim"),
    ("iot_hue_lightup", "iot_wemo_on"),
    ("iot_hue_lightdim", "iot_wemo_off"),
    ("iot_hue_lightdim", "iot_hue_lighton"),
    ("iot_wemo_off", "iot_hue_lighton"),
    ("iot_wemo_on", "iot_hue_lighton"),
    ("audio_volume_mute", "audio_volume_up"),
    ("audio_volume_mute", "audio_volume_other"),
    ("audio_volume_up", "audio_volume_down"),
    ("audio_volume_down", "audio_volume_other")
]


def intent_to_id(data_sb):
    intent2id = {}
    for elem in data_sb:
        intent = data_sb[elem]["intent"]
        if intent not in intent2id:
            intent2id[intent] = [elem]
        else:
            intent2id[intent].append(elem)

    return intent2id


def scenario_to_count(data):
    sce2count = {}
    for elem in data:
        scenario = data[elem]["scenario_1"]
        if scenario not in sce2count:
            sce2count[scenario] = 1
        else:
            sce2count[scenario] += 1
    
    return sce2count


def fileid_to_slurpid(slurp_data_train, slurp_data_dev, slurp_data_test):
    f_id_s_id = {}
    for elem in slurp_data_train:
        json_data = json.loads(elem)
        slurp_id = json_data["slurp_id"]
        for rec in json_data["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id
    for elem in slurp_data_dev:
        json_data = json.loads(elem)
        slurp_id = json_data["slurp_id"]
        for rec in json_data["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id
    for elem in slurp_data_test:
        json_data = json.loads(elem)
        slurp_id = json_data["slurp_id"]
        for rec in json_data["recordings"]:
            file_id = rec["file"]
            f_id_s_id[file_id] = slurp_id

    return f_id_s_id


def prepare_speechbrain(data):
    data_obj = {}
    for elem in data:
        json_elem = json.loads(elem)
        scenario = json_elem["scenario"]
        intent = json_elem["intent"]
        action = json_elem["action"]
        transcript = json_elem["sentence"]
        for rec in json_elem["recordings"]:
            file_id = rec["file"]
            data_path = os.path.join("/m/teamwork/t40511_asr/c/SLURP/audio/slurp_real/", file_id)

            data_obj[file_id] = {
                "data_path": data_path,
                "scenario": scenario,
                "intent": intent,
                "action": action,
                "transcript": transcript,
                "audio_len": 0
            }
    
    return data_obj


def create_cg_split(data_sb, intents, intent2id, n_samples):
    random.seed(2024)
    data_obj = {}

    for tuple_elem in intents:
        intent_1 = tuple_elem[0]
        intent_2 = tuple_elem[1]

        # get all the ids with the intents
        try:
            intent_1_ids = intent2id[intent_1]
            intent_2_ids = intent2id[intent_2]
        except:
            continue


        # sample n ids from each intent without repetition
        try:
            intent_1_samples = random.sample(intent_1_ids, n_samples)
        except:
            intent_1_samples = intent_1_ids
        try:
            intent_2_samples = random.sample(intent_2_ids, n_samples)
        except:
            intent_2_samples = intent_2_ids
        
        # create the object with those IDs
        for i in range(len(intent_1_samples)):
            for j in range(len(intent_2_samples)):
                first_data = data_sb[intent_1_samples[i]]
                second_data = data_sb[intent_2_samples[j]]
                
                data_path_1 = first_data["data_path"]
                scenario_1 = first_data["scenario"]
                intent_1 = first_data["intent"]
                action_1 = first_data["action"]
                transcript_1 = first_data["transcript"]
                # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "audio_len_1" from the data_obj)
                y, sr = librosa.load(data_path_1)
                audio_len_1 = librosa.get_duration(y=y, sr=sr)

                data_path_2 = second_data["data_path"]
                scenario_2 = second_data["scenario"]
                intent_2 = second_data["intent"]
                action_2 = second_data["action"]
                transcript_2 = second_data["transcript"]
                # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "audio_len_2" from the data_obj)
                y, sr = librosa.load(data_path_2)
                audio_len_2 = librosa.get_duration(y=y, sr=sr)
                
                data_obj[intent_1_samples[i] + "+" + intent_2_samples[j]] = {
                    "data_path_1": data_path_1,
                    "data_path_2": data_path_2,
                    "scenario_1": scenario_1,
                    "scenario_2": scenario_2,
                    "intent_1": intent_1,
                    "intent_2": intent_2,
                    "action_1": action_1,
                    "action_2": action_2,
                    "transcript_1": transcript_1,
                    "transcript_2": transcript_2,
                    "audio_len_1": audio_len_1,
                    "audio_len_2": audio_len_2
                }

    return data_obj


def create_non_cg_split(data, cg_data, sce2count, train_intents, test_intents, intent2id, n_samples):
    random.seed(2024)
    data_obj = {}
    # get half of the elements from the difficult split
    for tuple_elem in train_intents:
        intent_1 = tuple_elem[0]
        intent_2 = tuple_elem[1]

        # count the number of elements in cg_data that have the same intents
        elems_same_int = 0
        for elem in cg_data:
            if cg_data[elem]["intent_1"] == intent_1 and cg_data[elem]["intent_2"] == intent_2:
                elems_same_int += 1
        
        if elems_same_int == 0:
            continue
        
        # get all of the elements from cg_data that have the same intents
        elems_to_take = elems_same_int // 2
        for elem in cg_data:
            if cg_data[elem]["intent_1"] == intent_1 and cg_data[elem]["intent_2"] == intent_2:
                data_obj[elem] = cg_data[elem]
                elems_to_take -= 1
                if elems_to_take == 0:
                    break
        
    

    # get the other half of the elements from the data and make sure they are in test_intents
    sce2count_data_obj = scenario_to_count(data_obj)
    num_elems_missing = len(cg_data) - len(data_obj)
    for tuple_elem in test_intents:
        intent_1 = tuple_elem[0]
        intent_2 = tuple_elem[1]

        scenario = intent_1.split("_")[0]
        num_scenarios_in_cg_data = sce2count[scenario]

        # get all the ids with the intents
        try:
            intent_1_ids = intent2id[intent_1]
            intent_2_ids = intent2id[intent_2]
        except:
            continue

        # sample n ids from each intent without repetition
        try:
            intent_1_samples = random.sample(intent_1_ids, n_samples)
        except:
            intent_1_samples = intent_1_ids
        try:
            intent_2_samples = random.sample(intent_2_ids, n_samples)
        except:
            intent_2_samples = intent_2_ids
        
        for i in range(len(intent_1_samples)):
            if sce2count_data_obj[scenario] >= num_scenarios_in_cg_data:
                    break
            for j in range(len(intent_2_samples)):
                if sce2count_data_obj[scenario] >= num_scenarios_in_cg_data:
                    break
                
                first_data = data[intent_1_samples[i]]
                second_data = data[intent_2_samples[j]]
                
                data_path_1 = first_data["data_path"]
                scenario_1 = first_data["scenario"]
                intent_1 = first_data["intent"]
                action_1 = first_data["action"]
                transcript_1 = first_data["transcript"]
                # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "audio_len_1" from the data_obj)
                y, sr = librosa.load(data_path_1)
                audio_len_1 = librosa.get_duration(y=y, sr=sr)

                data_path_2 = second_data["data_path"]
                scenario_2 = second_data["scenario"]
                intent_2 = second_data["intent"]
                action_2 = second_data["action"]
                transcript_2 = second_data["transcript"]
                # if you do not need the audio length, you can remove the following two lines, which will speed up the process (also remember to remove "audio_len_2" from the data_obj)
                y, sr = librosa.load(data_path_2)
                audio_len_2 = librosa.get_duration(y=y, sr=sr)

                if intent_1 + "+" + intent_2 not in data_obj:
                    data_obj[intent_1_samples[i] + "+" + intent_2_samples[j]] = {
                        "data_path_1": data_path_1,
                        "data_path_2": data_path_2,
                        "scenario_1": scenario_1,
                        "scenario_2": scenario_2,
                        "intent_1": intent_1,
                        "intent_2": intent_2,
                        "action_1": action_1,
                        "action_2": action_2,
                        "transcript_1": transcript_1,
                        "transcript_2": transcript_2,
                        "audio_len_1": audio_len_1,
                        "audio_len_2": audio_len_2
                    }
                    sce2count_data_obj = scenario_to_count(data_obj)

    return data_obj


def save_cg_splits(train_obj_cg, dev_obj_cg, test_obj, f_id_s_id, save_path):
    # Save the splits as JSON files in a SpeechBrain format
    with open(os.path.join(save_path, "train_cg.json"), "w") as f:
        json.dump(train_obj_cg, f, indent=4)
    with open(os.path.join(save_path, "dev_cg.json"), "w") as f:
        json.dump(dev_obj_cg, f, indent=4)
    with open(os.path.join(save_path, "test.json"), "w") as f:
        json.dump(test_obj, f, indent=4)

    # Save the splits as ID, scenario CSV files
    with open(os.path.join(save_path, "train_cg.csv"), "w") as f:
        f.write("slurp_id_1,slurp_id_2,file_id,scenario\n")
        for elem in train_obj_cg:
            slurp_id_1 = str(f_id_s_id[elem.split("+")[0]])
            slurp_id_2 = str(f_id_s_id[elem.split("+")[1]])
            f.write(slurp_id_1 + "," + slurp_id_2 + "," + elem + "," + train_obj_cg[elem]["scenario_1"] + "\n")
    
    with open(os.path.join(save_path, "dev_cg.csv"), "w") as f:
        f.write("slurp_id_1,slurp_id_2,file_id,scenario\n")
        for elem in dev_obj_cg:
            slurp_id_1 = str(f_id_s_id[elem.split("+")[0]])
            slurp_id_2 = str(f_id_s_id[elem.split("+")[1]])
            f.write(slurp_id_1 + "," + slurp_id_2 + "," + elem + "," + dev_obj_cg[elem]["scenario_1"] + "\n")

    with open(os.path.join(save_path, "test.csv"), "w") as f:
        f.write("slurp_id_1,slurp_id_2,file_id,scenario\n")
        for elem in test_obj:
            slurp_id_1 = str(f_id_s_id[elem.split("+")[0]])
            slurp_id_2 = str(f_id_s_id[elem.split("+")[1]])
            f.write(slurp_id_1 + "," + slurp_id_2 + "," + elem + "," + test_obj[elem]["scenario_1"] + "\n")




def save_non_cg_splits(train_obj_non_cg, dev_obj_non_cg, f_id_s_id, save_path):
    # Save the splits as JSON files in a SpeechBrain format
    with open(os.path.join(save_path, "train_non_cg.json"), "w") as f:
        json.dump(train_obj_non_cg, f, indent=4)
    with open(os.path.join(save_path, "dev_non_cg.json"), "w") as f:
        json.dump(dev_obj_non_cg, f, indent=4)

    # Save the splits as ID, scenario CSV files
    with open(os.path.join(save_path, "train_non_cg.csv"), "w") as f:
        f.write("slurp_id_1,slurp_id_2,file_id,scenario\n")
        for elem in train_obj_non_cg:
            slurp_id_1 = str(f_id_s_id[elem.split("+")[0]])
            slurp_id_2 = str(f_id_s_id[elem.split("+")[1]])
            f.write(slurp_id_1 + "," + slurp_id_2 + "," + elem + "," + train_obj_non_cg[elem]["scenario_1"] + "\n")
    
    with open(os.path.join(save_path, "dev_non_cg.csv"), "w") as f:
        f.write("slurp_id_1,slurp_id_2,file_id,scenario\n")
        for elem in dev_obj_non_cg:
            slurp_id_1 = str(f_id_s_id[elem.split("+")[0]])
            slurp_id_2 = str(f_id_s_id[elem.split("+")[1]])
            f.write(slurp_id_1 + "," + slurp_id_2 + "," + elem + "," + dev_obj_non_cg[elem]["scenario_1"] + "\n")


if __name__ == "__main__":
    # Desired lengths for train, dev, and test: 36064, 6192, 3500
    original_slurp_path = "../../../slurp/dataset/slurp"
    save_path = "../../splits/DA_CG"

    with open(os.path.join(original_slurp_path, "train.jsonl"), "r") as f:
        train_data = f.readlines()
    with open(os.path.join(original_slurp_path, "devel.jsonl"), "r") as f:
        dev_data = f.readlines()
    with open(os.path.join(original_slurp_path, "test.jsonl"), "r") as f:
        test_data = f.readlines()

    # Get file_id to slurp_id mapping
    f_id_s_id = fileid_to_slurpid(train_data, dev_data, test_data)
    
    # Prepare data in a speechbrain format
    train_data_sb = prepare_speechbrain(train_data)
    dev_data_sb = prepare_speechbrain(dev_data)
    test_data_sb = prepare_speechbrain(test_data)

    # Get intent2id
    intent2id_train = intent_to_id(train_data_sb)
    intent2id_dev = intent_to_id(dev_data_sb)
    intent2id_test = intent_to_id(test_data_sb)

    # Create the CG splits
    train_obj_cg = create_cg_split(train_data_sb, train_intents, intent2id_train, 28)
    dev_obj_cg = create_cg_split(dev_data_sb, train_intents, intent2id_dev, 12)
    test_obj = create_cg_split(test_data_sb, test_intents, intent2id_test, 10)

    # Save the CG splits
    save_cg_splits(train_obj_cg, dev_obj_cg, test_obj, f_id_s_id, save_path)


    # Create the non-CG splits
    sce2count_train = scenario_to_count(train_obj_cg)
    sce2count_dev = scenario_to_count(dev_obj_cg)
    
    train_obj_non_cg = create_non_cg_split(train_data_sb, train_obj_cg, sce2count_train, train_intents, test_intents, intent2id_train, 28)
    dev_obj_non_cg = create_non_cg_split(dev_data_sb, dev_obj_cg, sce2count_dev, train_intents, test_intents, intent2id_dev, 15)

    # Save the non-CG splits
    save_non_cg_splits(train_obj_non_cg, dev_obj_non_cg, f_id_s_id, save_path)
