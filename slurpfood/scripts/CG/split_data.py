import json
import torch
import os
import sys
from tqdm import tqdm
import random
random.seed(42)
from dbca.freq_mats import save_struct

from data_analysis import divergence
import argparse

AUDIO_PATH = "../../../slurp/scripts/audio/slurp_real"
# ORIGINAL_DATA_PATH = "/m/teamwork/t40511_asr/c/SLURP/speechbrain_data/original_splits"
ORIGINAL_DATA_PATH = None
METADATA_DIR = "../../../slurp/dataset/slurp"

def orig_test_set_ids():
    """Get the ids of the original test set."""
    ids  = []
    with open(f"{METADATA_DIR}/test.json", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            ids.append(line['slurp_id'])
    return ids


def parse_slurp_metadata(sample_md, disregard_order=False):
    """Parse the metadata of a SLURP sample.
    
    sample_md input format:
    {
        "slurp_id": 13804,
        "sentence": "siri what is one american dollar in japanese yen",
        "sentence_annotation": "siri what is one [currency_name : american dollar] in [currency_name : japanese yen]",
        "intent": "qa_currency",
        "action": "currency",
        "tokens": [
            {"surface": "siri", "id": 0, "lemma": "siri", "pos": "VB"},
            {"surface": "what", "id": 1, "lemma": "what", "pos": "WP"},
            {"surface": "is", "id": 2, "lemma": "be", "pos": "VBZ"},
            {"surface": "one", "id": 3, "lemma": "one", "pos": "CD"},
            {"surface": "american", "id": 4, "lemma": "american", "pos": "JJ"},
            {"surface": "dollar", "id": 5, "lemma": "dollar", "pos": "NN"},
            {"surface": "in", "id": 6, "lemma": "in", "pos": "IN"},
            {"surface": "japanese", "id": 7, "lemma": "japanese", "pos": "JJ"},
            {"surface": "yen", "id": 8, "lemma": "yen", "pos": "NNS"}
        ],
        "scenario": "qa",
        "recordings": [
            {"file": "audio-1434542201-headset.flac", "wer": 0.0, "ent_wer": 0.0, "status": "correct"},
            {"file": "audio-1434542201.flac", "wer": 0.0, "ent_wer": 0.0, "status": "correct"}
        ],
        "entities": [
            {"span": [4, 5], "type": "currency_name"},
            {"span": [7, 8], "type": "currency_name"}
        ]
    }
    
    Output format:
    {
        "slurp_id": 13804,
        'recordings': ["audio-1434542201-headset.flac", "audio-1434542201.flac"],
        'atom:action': "currency",
        'atom:scenario': "qa",
        'atom:entity_types': ["currency_name", "currency_name"],
        'atom:entity_surfs': ["american dollar", "japanese yen"],
        'compound:entity_type_combination': "currency_name+currency_name",
        'compound:entity_surfs': "american dollar+japanese yen",
        'compound:scenario_action': "qa+currency",
        'compound:scenario_entity_types': "qa+currency_name+currency_name",
        'compound:action_entity_types': "currency+currency_name+currency_name",
        'compound:scenario_action_entity_types': "qa+currency+currency_name+currency_name"
    }
    """

    parsed = {}

    # atoms:
    for label_type in ['atom:action', 'atom:scenario']:
        orig_label = sample_md[label_type.split(':')[1]]
        parsed[label_type] = orig_label

    parsed['atom:entity_types'] = [ent['type'] for ent in sample_md['entities']]

    parsed['atom:entity_surfs'] = []
    idx2token = {token['id']: token['surface'] for token in sample_md['tokens']}
    for ent in sample_md['entities']:
        parsed['atom:entity_surfs'].append(' '.join([idx2token[idx] for idx in ent['span']]))

    # compounds:
    sep = '+'
    if disregard_order:
        parsed['atom:entity_types'].sort()
    entity_type_comb = sep.join(parsed['atom:entity_types'])

    parsed['compound:entity_type_combination'] = entity_type_comb
    parsed['compound:entity_surfs'] = sep.join(parsed['atom:entity_surfs'])
    parsed['compound:scenario_action'] = f"{parsed['atom:scenario']}{sep}{parsed['atom:action']}"
    parsed['compound:scenario_entity_types'] = f"{parsed['atom:scenario']}{sep}{entity_type_comb}"
    parsed['compound:action_entity_types'] = f"{parsed['atom:action']}{sep}{entity_type_comb}"
    parsed['compound:scenario_action_entity_types'] = \
        f"{parsed['atom:scenario']}{sep}{parsed['atom:action']}{sep}{entity_type_comb}"

    parsed['recordings'] = [rec['file'] for rec in sample_md['recordings']]
    parsed['slurp_id'] = sample_md['slurp_id']

    return parsed


def atoms_coms_recs_per_sentence(data, com_type, disregard_order=False):
    """Get the atoms, compounds and recordings per sentence. atom_type defines the type of
    atom to be used. Compound is always the combination of atoms.
    """
    atoms_per_sentence = {}
    compounds_per_sentence = {}
    recs_per_sentence = {}

    for sample_metadata in data.values():
        parsed_elem = parse_slurp_metadata(sample_metadata, disregard_order)
        atoms_per_sentence[parsed_elem['slurp_id']] = parsed_elem[com_type].split('+')
        compounds_per_sentence[parsed_elem['slurp_id']] = [parsed_elem[com_type]]
        recs_per_sentence[parsed_elem['slurp_id']] = parsed_elem['recordings']

    return atoms_per_sentence, compounds_per_sentence, recs_per_sentence


def freq_matrices(atoms_per_sent, compounds_per_sent, recs_per_sent, weight_by_num_recs=False):
    """Create frequency matrices of atoms and compounds per sentence.
    
    Parameters
    ----------
    atoms_per_sents : dict
        Dictionary with sentence_ids as keys and atoms as values.
    compounds_per_sents : dict
        Dictionary with sentence_ids as keys and compounds as values.
        
    Returns
    -------
    atom_matrix : pytorch tensor
        Matrix with sentences as rows and atom types as columns. Each cell contains the frequency
        of the atom type in the sentence.
    compound_matrix : pytorch tensor
        Matrix with sentences as rows and compound types as columns. Each cell contains
        the frequency of the compound type in the sentence.
    ids2rows : dict
        Dictionary with sentence ids as keys and row indices as values.
    recs_per_used_sents : list
        List with the number of recordings per sentence.
    atom2col : dict
        Dictionary with atom types as keys and column indices as values.
    compound2col : dict
        Dictionary with compound types as keys and column indices as values.
    """

    # create dictionary with all atom types
    atom_types = set()
    for sentence_id in atoms_per_sent:
        for atom in atoms_per_sent[sentence_id]:
            atom_types.add(atom)
    atom_types = list(atom_types)
    atom_types.sort()
    atom2col = {atom: idx for idx, atom in enumerate(atom_types)}

    # create dictionary with all compound types
    compound_types = set()
    for sentence_id in compounds_per_sent:
        for compound in compounds_per_sent[sentence_id]:
            compound_types.add(compound)
    compound_types = list(compound_types)
    compound_types.sort()

    compound2col = {compound: idx for idx, compound in enumerate(compound_types)}

    # create frequency matrices
    num_rows = len(atoms_per_sent)
    atom_matrix = torch.zeros(num_rows, len(atom_types), dtype=torch.uint8)
    compound_matrix = torch.zeros(num_rows, len(compound_types), dtype=torch.uint8)
    ids2rows = {}
    recs_per_used_sents = []
    # if there are atoms, there are also compounds
    for idx, sentence_id in enumerate(atoms_per_sent):
        ids2rows[sentence_id] = idx
        num_recs = len(recs_per_sent[sentence_id])
        recs_per_used_sents.append(str(num_recs))
        if weight_by_num_recs:
            increment = num_recs
        else:
            increment = 1
        for atom in atoms_per_sent[sentence_id]:
            atom_matrix[idx, atom2col[atom]] += increment
        if sentence_id in compounds_per_sent:
            for compound in compounds_per_sent[sentence_id]:
                compound_matrix[idx, compound2col[compound]] += increment

    return atom_matrix.to_sparse(), compound_matrix.to_sparse(), ids2rows, \
        recs_per_used_sents, atom2col, compound2col


def save_prepped_data(output_dir, atom_m, com_m, sent_ids, sent_sizes, atom_ids, com_ids):
    """Save the prepared data to disk."""
    atom_freq_file = os.path.join(output_dir, 'atom_freqs.pt')
    com_freq_file = os.path.join(output_dir, 'compound_freqs.pt')
    torch.save(atom_m, atom_freq_file)
    torch.save(com_m, com_freq_file)
    save_struct(sent_ids, os.path.join(output_dir, 'used_sent_ids.txt'), overwrite=True)
    save_struct(sent_sizes, os.path.join(output_dir, 'sent_sizes.txt'), overwrite=True)
    save_struct(atom_ids, os.path.join(output_dir, 'atom_ids.pkl'), overwrite=True)
    save_struct(com_ids, os.path.join(output_dir, 'com_ids.pkl'), overwrite=True)


def id_to_length(data_path=None):
    """Create a dictionary with mapping from recording ID (e.g. audio-1501407267.flac)
    to its length in seconds."""
    if data_path:
        id2length = {}
        for subset in ['train', 'dev', 'test']:
            with open(os.path.join(data_path, f"{subset}.json"), "r", encoding="utf-8") as f:
                dataset = json.load(f)

            for elem in dataset:
                id2length[elem] = dataset[elem]["length"]

    if not data_path:
        import pickle as pkl

        if os.path.exists(AUDIO_PATH + '/id2length.pkl'):
            with open(AUDIO_PATH + '/id2length.pkl', 'rb') as f:
                id2length = pkl.load(f)
            return id2length

        import glob
        import librosa
        id2length = {}
        for file in tqdm(glob.glob(AUDIO_PATH + '/*.flac'), desc='Reading lengths of audio files'):
            y, sr = librosa.load(file)
            length = librosa.get_duration(y=y, sr=sr)
            id2length[os.path.basename(file)] = length

        with open(AUDIO_PATH + '/id2length.pkl', 'wb') as f:
            pkl.dump(id2length, f)

    return id2length


def slurp2speechbrain(slurp_metadata, slurp_ids, id2length, output_file):
    """Convert the SLURP metadata to the SpeechBrain format.
    
    slurp_ids is a list of SLURP IDs to be included in the output file.
    
    id2length is a dictionary with the mapping from the recording ID (e.g. audio-1501407267.flac)
    to its length in seconds.

    Input format is
    1501407267: {
        "slurp_id": 1501407267,
        "sentence": "this is a sentence",
        "scenario": "calendar",
        "action": "set",
        "intent": "calendar_set",
        "recordings": [
            {
                "file": "audio-1501407267.flac"
            },
        ]
    }
    
    Output format is
    {
        "audio-1501407267.flac": {
            "data_path": "/m/teamwork/t40511_asr/c/SLURP/audio/slurp_real/audio-1501407267.flac",
            "transcript": "this is a sentence",
            "scenarion": "calendar",
            "action": "set",
            "intent": "calendar_set",
            "length": 1.880907029478458
        },
    }
    
    """
    output_dict = {}
    for slurp_id in slurp_ids:
        sample = slurp_metadata[slurp_id]
        for recording in sample["recordings"]:
            rec_id = recording["file"]
            output_dict[rec_id] = {
                "data_path": os.path.join(AUDIO_PATH, rec_id),
                "transcript": sample["sentence"],
                "scenario": sample["scenario"],
                "action": sample["action"],
                "intent": sample["intent"],
                "length": id2length[rec_id]
            }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=4)


def label_distributions(data, disregard_order=False, weight_by_num_recs=False, two_actions=False):
    """Get the distributions of scenarios, actions and entity types as well as
    the distributions of the combinations ("compounds") of these labels."""

    if two_actions:
        all_distribs = {'atom:scenario': {}, 'atom:action': {}, 'compound:scenario_actions': {}}
        # print(data)
        for sample_md in data:
            for label_type, label_counts in all_distribs.items():
                if isinstance(sample_md[label_type], list):
                    for l in sample_md[label_type]:
                        if l not in label_counts:
                            all_distribs[label_type][l] = 1
                        else:
                            all_distribs[label_type][l] += 1
                else:
                    if sample_md[label_type] not in label_counts:
                        label_counts[sample_md[label_type]] = 1
                    else:
                        label_counts[sample_md[label_type]] += 1
    else:
        all_distribs = {
            'atom:scenario': {},
            'atom:action': {},
            'atom:entity_types': {},
            'compound:entity_type_combination': {},
            'compound:scenario_action': {},
            'compound:scenario_entity_types': {},
            'compound:action_entity_types': {},
            'compound:scenario_action_entity_types': {},
        }

        for sample_metadata in data.values():
            parsed_md = parse_slurp_metadata(sample_metadata, disregard_order)
            if weight_by_num_recs:
                increment = len(parsed_md['recordings'])
            else:
                increment = 1

            for label_type, label_counts in all_distribs.items():
                label = parsed_md[label_type]
                if isinstance(label, list):
                    for l in label:
                        if l not in label_counts:
                            label_counts[l] = increment
                        else:
                            label_counts[l] += increment
                else:
                    if label not in label_counts:
                        label_counts[label] = increment
                    else:
                        label_counts[label] += increment

    return all_distribs


def distribution_similarity(train, dev, test, metadata, verbose=False, weight_by_num_recs=False, two_actions=False):
    """
    Calculate the divergence between the label distributions of
    the train, dev and test sets.
    """

    print()
    print("Number of slurp_ids in train set:", len(train))
    print("Number of slurp_ids in dev set:", len(dev))
    print("Number of slurp_ids in test set:", len(test))
    print()

    if not two_actions:
        for subset_name, subset in [('train', train), ('dev', dev), ('test', test)]: 
            print(f'Number of recordings in {subset_name} set:',
                sum([len(annotations['recordings']) for annotations in subset.values()]))
            print(f'Number of original recordings associated with {subset_name} set slurp_ids:',
                len(set([rec['file'] for slurpid in subset.keys()
                                    for rec in metadata[slurpid]['recordings']])))

        print()
        print("Train-test slurp_id overlap:", len(set(train.keys()) & set(test.keys())))
        print("Train-dev slurp_id overlap:", len(set(train.keys()) & set(dev.keys())))
        print("Dev-test slurp_id overlap:", len(set(dev.keys()) & set(test.keys())))
        print()

    train_distribs = label_distributions(train, weight_by_num_recs=weight_by_num_recs, two_actions=two_actions)
    dev_distribs = label_distributions(dev, weight_by_num_recs=weight_by_num_recs, two_actions=two_actions)
    test_distribs = label_distributions(test, weight_by_num_recs=weight_by_num_recs, two_actions=two_actions)

    label_types = list(train_distribs.keys())

    all_labels = {}
    label_ids = {}
    for label_type in label_types:
    # for label_type in ['atom:scenario', 'atom:action', 'compound:scenario_action']:
        all_labels[label_type] = set(train_distribs[label_type].keys()) \
                               | set(dev_distribs[label_type].keys()) \
                               | set(test_distribs[label_type].keys())
        label_ids[label_type] = sorted(list(all_labels[label_type]))

        print(f'Label type: {label_type}')

        train_distr = [train_distribs[label_type].get(l, 0) for l in label_ids[label_type]]
        test_distr = [test_distribs[label_type].get(l, 0) for l in label_ids[label_type]]
        dev_distr = [dev_distribs[label_type].get(l, 0) for l in label_ids[label_type]]

        
        if verbose:
            print(f"\tAll {label_type}s: ", label_ids[label_type])
            #  print distritbuions as table
            print(f"\t{'':<100}{'train':<10}{'dev':<5}{'test':<5}")
            for i, label in enumerate(label_ids[label_type]):
                print(f"\t{label:<100}{train_distr[i]:<8}{dev_distr[i]:<8}{test_distr[i]:<8}")

        if label_type.startswith('atom'):
            alpha=0.5
        elif label_type.startswith('compound'):
            alpha=0.1

        print(f"\tTrain-test divergence: {divergence(train_distr, test_distr, alpha=alpha)} " + \
              f"(similarity: {round(1 - divergence(train_distr, test_distr, alpha=alpha), 2)})")
        # print(f"\tTrain-dev divergence: {divergence(train_distr, dev_distr, alpha=alpha)} " + \
            # f"(similarity: {round(1 - divergence(train_distr, dev_distr, alpha=alpha), 2)})")
        print(f"\tDev-test divergence: {divergence(dev_distr, test_distr, alpha=alpha)} " + \
                f"(similarity: {round(1 - divergence(dev_distr, test_distr, alpha=alpha), 2)})")
        print()

        # overlap of the label types
        tot_num_labels = len(all_labels[label_type])
        train_labels = set(train_distribs[label_type].keys())
        test_labels = set(test_distribs[label_type].keys())
        dev_labels = set(dev_distribs[label_type].keys())
        print(f"\tTotal number of {label_type} types:", tot_num_labels)
        print(f"\tNumber of {label_type} types in train set:", len(train_labels))
        print(f"\tNumber of {label_type} types in test set:", len(test_labels))
        print(f"\tNumber of {label_type} types in dev set:", len(dev_labels))
        print("\tTrain-test overlap:", len(train_labels & test_labels))
        print("\tTrain-dev overlap:", len(train_labels & dev_labels))
        print("\tDev-test overlap:", len(dev_labels & test_labels))

        print("\tLables that are in training but not in test:", train_labels.difference(test_labels))
        print("\tLables that are in test but not in training:", test_labels.difference(train_labels))
        # print()
        
        # if two_actions:
        #     train_set_actions = set()
        #     for sample_md in train:
        #         for a in sample_md['atom:action'].split('+'):
        #             train_set_actions.add(a)
        #     print('train_set_actions:', train_set_actions)
        #     test_set_actions = set()
        #     for sample_md in test:
        #         for a in sample_md['atom:action'].split('+'):
        #             test_set_actions.add(a)
        #     print('test_set_actions:', test_set_actions)
        #     print('train_set_actions - test_set_actions:', train_set_actions - test_set_actions)
        #     print('test_set_actions - train_set_actions:', test_set_actions - train_set_actions)
        #     print()
            
        

def read_ids(input_file):
    """ read integers in first column in txt file """
    with open(input_file, "r", encoding="utf-8") as f:
        ids = [int(line.split()[0]) for line in f.readlines()]
    return ids


def make_rec2slurpid_dict(metadata):
    rec2slurpid = {}
    for slurpid in metadata:
        for rec in metadata[slurpid]['recordings']:
            rec2slurpid[rec['file']] = slurpid
    return rec2slurpid


def recs_to_original_metadata(sb_file, metadata, rec2slurpid, two_actions=False):
    """Create a new metadata dictionary from the SpeechBrain format."""

    with open(sb_file, "r", encoding='utf-8') as f:
        jsondict = json.load(f)

    new_metadata = {}
    for rec_file in jsondict:
        if two_actions:
            rec_file = rec_file.split('+')
        else:
            rec_file = [rec_file]
        for rec in rec_file:
            slurpid = rec2slurpid[rec]
            if slurpid not in new_metadata:
                new_metadata[slurpid] = metadata[slurpid]
                new_metadata[slurpid]['recordings'] = [{'file': rec}]
            else:
                new_metadata[slurpid]['recordings'].append({'file': rec})
    return new_metadata


def parse_two_action_sb_metadata(sb_file):

    with open(sb_file, "r", encoding='utf-8') as f:
        jsondict = json.load(f)

    all_parsed = []
    for md in jsondict.values():
        parsed = {}
        # parsed['atom:action'] = f"{md['action_1']}+{md['action_2']}"
        parsed['atom:action'] = [md['action_1'], md['action_2']]
        # parsed['atom:scenario'] = f"{md['scenario_1']}+{md['scenario_2']}"
        parsed['atom:scenario'] = md['scenario_1']
        parsed['compound:scenario_actions'] = \
            f"{md['scenario_1']}+{md['action_1']}+{md['action_2']}"
        all_parsed.append(parsed)

    return all_parsed


def get_orig_metadata():
    metadata = {}
    for subset in ['train', 'devel', 'test']:
        with open(f"{METADATA_DIR}/{subset}.jsonl", "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                metadata[line["slurp_id"]] = line
    return metadata


def metadata_from_id_files(sample_ids_files, metadata):
    """Get the metadata for the samples in the sample_ids_files."""
    if len(sample_ids_files) == 3:
        tr_dv_te = []
        for subset in sample_ids_files:
            tr_dv_te.append({id: metadata[id] for id in read_ids(subset)})
    elif len(sample_ids_files) == 2:
        tr_dv_te = [{},{},{}]
        tr_dv_te[0] = {id: metadata[id] for id in read_ids(sample_ids_files[0])}
        tr_dv_te[2] = {id: metadata[id] for id in read_ids(sample_ids_files[1])}
    else:
        raise ValueError('Please specify either 2 or 3 splitted_data_ids.')

    return tr_dv_te

def remove_samples_not_in_train_set(metadata, sb_data_dir, id2length_file, two_actions=False):
    rec2slurpid = make_rec2slurpid_dict(metadata)
    tr_dv_te = []
    for subset_name in ['train', 'dev', 'test']:
        filepath = os.path.join(sb_data_dir, f'{subset_name}.json')
        if not two_actions:
            tr_dv_te.append(recs_to_original_metadata(filepath, metadata, rec2slurpid))
        else:
            tr_dv_te.append(parse_two_action_sb_metadata(filepath))

    train_set, _, test_set = tr_dv_te
    remove_ids_test = []
    if not two_actions:
        trainset_labels = {}
        trainset_labels['action'] = set()
        for sid, sample_md in train_set.items():
            trainset_actions.add(sample_md['action'])
        trainset_labels['scenario'] = set()
        for sid, sample_md in train_set.items():
            trainset_scenarios.add(sample_md['scenario'])

        new_test_set = {}
        for label in ['atom:action', 'atom:scenario']:
            for idx, sid in enumerate(test_set):
                if test_set[sid][label] in trainset_labels[label]:
                    new_test_set[sid] = test_set[sid]
                else:
                    print(f"Removing sample {sid} with {label} {test_set[sid][label]} not in train set.")
                    remove_ids_test.append(idx)
    else:
        trainset_actions = set()
        for sample_md in train_set:
            for a in sample_md['atom:action'].split('+'):
                trainset_actions.add(a)

        trainset_scenarios = set()
        for sample_md in train_set:
            for s in sample_md['atom:scenario'].split('+'):
                trainset_scenarios.add(s)
        
        print('trainset_actions:', trainset_actions)
        print('trainset_scenarios:', trainset_scenarios)
        
        new_test_set = {}
        for idx, sample_md in enumerate(test_set):
            if sample_md['atom:action'].split('+')[0] in trainset_actions and \
                sample_md['atom:action'].split('+')[1] in trainset_actions and \
                sample_md['atom:scenario'].split('+')[0] in trainset_scenarios:
                new_test_set[idx] = sample_md
            else:
                # print(f"Removing sample {sample_md} not in train set.")
                remove_ids_test.append(idx)

    test_set = new_test_set

    print(f'Number of samples in test set after removing samples with not in train set: {len(test_set)}')
    print(f'Number of samples to remove from test set: {len(remove_ids_test)}')

    slurp2speechbrain(metadata, test_set, id2length_file,
                    os.path.join(sb_data_dir, "test_removed_samples.json"))


def main(args):
    metadata = get_orig_metadata()

    if args.prepare_data:
        if args.prepped_data_dir is None:
            raise ValueError('Please specify prepped_data_dir.')
        if not os.path.exists(args.prepped_data_dir):
            os.makedirs(args.prepped_data_dir)

        parsed_data_tuple = freq_matrices(*atoms_coms_recs_per_sentence(metadata,
                                                f'compound:{args.compound_type}',
                                                disregard_order=False))

        print(f'Number of atom types: {len(parsed_data_tuple[4])}, '
            f'Number of compound types: {len(parsed_data_tuple[5])}')
        save_prepped_data(args.prepped_data_dir, *parsed_data_tuple)

    if args.split_data:
        from dbca.divide import FromEmptySets

        if args.splitted_data_dir is None or args.prepped_data_dir is None:
            raise ValueError('Please specify splitted_data_dir and prepped_data_dir.')
        if not os.path.exists(args.splitted_data_dir):
            os.makedirs(args.splitted_data_dir)
        else:
            print(f'Output directory {args.splitted_data_dir} already exists.')
            sys.exit()

        divide_train_test = FromEmptySets(
            data_dir=args.prepped_data_dir,
            subsample_size=100,
            subsample_iter=1,
            presplit_train_test=args.from_presplit,
        )

        if args.from_presplit is None:
            min_test_percent = 0.1
            max_test_percent = 0.2
        else:
            min_test_percent = 0.0
            max_test_percent = 0.0

        divide_train_test.divide_corpus(
            target_atom_div=0.0,
            target_com_div=args.comdiv,
            min_test_percent=min_test_percent,
            max_test_percent=max_test_percent,
            # select_n_samples=use_n_samples,
            print_every=1000,
            # max_iters=args.max_iters,
            save_cp=500,
            output_dir=args.splitted_data_dir,
            # move_a_sample_iter=args.move_a_sample_iter,
        )

    id2length_file = id_to_length(ORIGINAL_DATA_PATH)

    if args.prepare_sb_data:
        if args.splitted_data_ids is None or args.sb_data_dir is None:
            raise ValueError('Please specify splitted_data_ids and sb_data_dir.')
        if not os.path.exists(args.sb_data_dir):
            os.makedirs(args.sb_data_dir)
        else:
            print(f'Output directory {args.sb_data_dir} already exists. Exiting.')
            sys.exit()

        rest_of_train_set = list(set(metadata.keys()).difference(
                            set(read_ids(f'{args.prepped_data_dir}/used_sent_ids.txt'))))

        train_set = rest_of_train_set + read_ids(args.splitted_data_ids[0])
        test_set = read_ids(args.splitted_data_ids[1])

        # select random dev set indices
        random.shuffle(train_set)
        num_train = int(len(train_set) * args.train_dev_split)
        train_set_new = train_set[:num_train]
        dev_set = train_set[num_train:]

        # write ids to file
        if not os.path.exists(args.sb_data_dir):
            os.makedirs(args.sb_data_dir)
        with open(os.path.join(args.sb_data_dir, "train.txt"), "w", encoding="utf-8") as f:
            for id in train_set_new:
                f.write(f"{id}\n")
        with open(os.path.join(args.sb_data_dir, "test.txt"), "w", encoding="utf-8") as f:
            for id in test_set:
                f.write(f"{id}\n")
        with open(os.path.join(args.sb_data_dir, "dev.txt"), "w", encoding="utf-8") as f:
            for id in dev_set:
                f.write(f"{id}\n")

        slurp2speechbrain(metadata, train_set_new, id2length_file,
                        os.path.join(args.sb_data_dir, "train.json"))
        slurp2speechbrain(metadata, dev_set, id2length_file,
                        os.path.join(args.sb_data_dir, "dev.json"))
        slurp2speechbrain(metadata, test_set, id2length_file,
                        os.path.join(args.sb_data_dir, "test.json"))


    if args.analyse_splits:
        if args.sb_data_dir and os.path.isdir(args.sb_data_dir):
            print(f'\nAnalyse splits in {args.sb_data_dir}')

            # check if there is train.txt in args.sb_data_dir
            if os.path.exists(os.path.join(args.sb_data_dir, "train.txt")) and \
                os.path.exists(os.path.join(args.sb_data_dir, "dev.txt")) and \
                os.path.exists(os.path.join(args.sb_data_dir, "test.txt")):
                tr_dv_te = []
                for subset_name in ['train', 'dev', 'test']:
                    filepath = os.path.join(args.sb_data_dir, f'{subset_name}.txt')
                    tr_dv_te.append({id: metadata[id] for id in read_ids(filepath)})
            else:
                rec2slurpid = make_rec2slurpid_dict(metadata)
                tr_dv_te = []
                for subset_name in ['train', 'dev', 'test']:
                    filepath = os.path.join(args.sb_data_dir, f'{subset_name}.json')
                    if not args.two_actions:
                        tr_dv_te.append(recs_to_original_metadata(filepath, metadata, rec2slurpid))
                    else:
                        tr_dv_te.append(parse_two_action_sb_metadata(filepath))
        elif args.splitted_data_ids is not None:
            print(f'\nAnalyse splits in {args.splitted_data_ids}')
            tr_dv_te = metadata_from_id_files(args.splitted_data_ids, metadata)
        else:
            raise ValueError('Please specify either sb_data_dir or splitted_data_ids.')

        distribution_similarity(*tr_dv_te, metadata, verbose=args.verbose,
                                weight_by_num_recs=args.weight_by_num_recs,
                                two_actions=args.two_actions )

    # if args.remove_samples_not_in_train_set:
    #     remove_samples_not_in_train_set(metadata, args.sb_data_dir, id2length_file, args.two_actions)
    if args.remove_samples_not_in_train_set:
        with open("remove_from_dbca_diff.txt", "r", encoding="utf-8") as f:
            slurp_ids = [int(line.strip()) for line in f.readlines() if line.strip()]

        rec2slurpid = make_rec2slurpid_dict(metadata)
        tr_dv_te = []
        filepath = os.path.join(args.sb_data_dir, 'test_old.json')
        original_test_set = recs_to_original_metadata(filepath, metadata, rec2slurpid)
        with open(filepath, "r", encoding="utf-8") as f:
            test_set = json.load(f)

        # get the idxs in test_set that correspond to slurp_ids

        remove_idxs = []
        for i, rec in enumerate(test_set):
            if original_test_set[rec2slurpid[rec]]['slurp_id'] in slurp_ids:
                remove_idxs.append(i)

        with open("remove_idxs.txt", "w", encoding="utf-8") as f:
            for idx in remove_idxs:
                f.write(f"{idx}\n")



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--prepare-data', action='store_true')
    args.add_argument('--prepped-data-dir', type=str)
    args.add_argument('--compound-type', type=str, default='scenario_entity_types')

    args.add_argument('--split-data', action='store_true')
    args.add_argument('--comdiv', type=float, default=1.0)
    args.add_argument('--splitted-data-dir', type=str,)
    args.add_argument('--from-presplit', type=str,
        help='Path to directory with train.txt, dev.txt and test.txt')

    args.add_argument('--prepare-sb-data', action='store_true')
    args.add_argument('--splitted-data-ids', nargs='+', type=str)
    args.add_argument('--train-dev-split', type=float, default=0.9)
    args.add_argument('--sb-data-dir', type=str,)
    args.add_argument('--two-actions', action='store_true')

    args.add_argument('--analyse-splits', action='store_true')
    args.add_argument('--verbose', action='store_true')
    args.add_argument('--weight-by-num-recs', action='store_true')

    args.add_argument('--remove-samples-not-in-train-set', action='store_true')
    args = args.parse_args()

    main(args)
