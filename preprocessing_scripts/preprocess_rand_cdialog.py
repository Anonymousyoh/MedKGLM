from transformers import AutoTokenizer
import torch
import os
import codecs
import json
import glob
import random
import csv
import argparse
from pytorch_pretrained_bert import BertTokenizer
from transformers import RobertaTokenizer


def match(sent1, sent2):
    sent1 = sent1[8:].split()
    sent2 = sent2.split()
    # print('ss1',sent1)
    # print('ss2',sent2)

    common = set(sent1).intersection(set(sent2))
    # print('c',common)
    # print(len(common)/(len(set(sent1))))
    # print("score",len(common) / len(set(sent1)))
    # print("\n")
    if len(common) / len(set(sent1)) > 0.90:
        # print('True')
        return True
    else:
        return False


def clean_dataset(dataset_file, json_file):
    rows = []
    f_in = open(dataset_file, "r", encoding='cp1252')
    # with open(dataset_file, "r", encoding='utf-8', errors='ignore') as file:
    csvreader = csv.reader(f_in)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

    f_json = open(json_file, "w", encoding='utf-8')
    # row[1] = conv_id
    # row[2] = speaker
    # row[3] = turn_num
    # row[5] = utterance
    # row[6] = entity

    total = 0
    last_turn = 0
    last_dialog = {}
    last_list = []
    Dialog_list = []
    id = "1"
    sen = ""
    print(len(rows))
    for lst in rows:
        if lst[1] != id:
            last_dialog["Turn"] = int(int(last_turn) / 2)
            last_dialog["Id"] = int(id)
            last_dialog["Dialogue"] = last_list[:]
            Dialog_list.append(last_dialog.copy())
            last_list.clear()
        id = lst[1]
        last_turn = lst[3]
        sen = lst[2].strip() + ": " + lst[5].strip()
        sen = sen.strip()
        last_list.append(sen)
        # print(len(Dialog_list))
    # print(len(Dialog_list))
    # print(Dialog_list[1])
    last_dialog["Turn"] = int(int(last_turn) / 2)
    last_dialog["Id"] = int(id)
    last_dialog["Dialogue"] = last_list[:]
    Dialog_list.append(last_dialog.copy())

    # print(Dialog_list[0])
    # print(last_list)

    print(len(Dialog_list))

    # print("Total Cases: ", json_file.split('.')[0].split('/')[-1], id)
    json.dump(Dialog_list, f_json, indent=4)
    f_in.close()
    f_json.close()
    return id


def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    for source_seq in source_seqs:
        # 去掉 xx：
        # print('sss',source_seq[8:])
        encoder_input += (source_seq[8:].split(' ')) + ["</s>"]

    decoder_input = ["<s>"] + (target_seq[7:].split(' '))  # 去掉 xx：
    # print(encoder_input)
    # print(decoder_input)

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "</s>" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("</s>", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    ee = encoder_input = ["<s>"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["</s>"]

    # print(encoder_input)

    # conver to ids
    encoder_input = tokenizer((' ').join(encoder_input))['input_ids'][1:-1][:MAX_ENCODER_SIZE - 1] + [
        tokenizer((' ').join(encoder_input))['input_ids'][-1]]  # tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer((' ').join(decoder_input))['input_ids'][1:-1][:MAX_DECODER_SIZE - 1] + [
        tokenizer((' ').join(decoder_input))['input_ids'][-1]]  # tokenizer.convert_tokens_to_ids(decoder_input)

    enc_len = len(encoder_input)
    dec_len = len(decoder_input)

    # print(enc_len)
    if enc_len > 400:
        print(encoder_input)
        print(len(ee))
        print(ee)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    global tot, maxi, mini

    tot = tot + len(encoder_input) + len(decoder_input)
    maxi = max(maxi, len(encoder_input))
    maxi = max(maxi, len(decoder_input))

    mini = min(mini, len(encoder_input))
    mini = min(mini, len(decoder_input))

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # print('encoder_input',mask_encoder_input)
    # print('decoder_input',mask_decoder_input)

    # print(pp)

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)

    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input


def make_dataset(data, file_name):
    train_data = []
    count = 0
    for d in data:
        # print(count)
        d_len = len(d)
        for i in range(d_len // 2):
            # print('src', d[:2 * i + 1])
            # print('trg', d[2 * i + 1])

            encoder_input, decoder_input, mask_encoder_input, mask_decoder_input = seq2token_ids(d[:2 * i + 1],
                                                                                                 d[2 * i + 1])
            train_data.append((encoder_input,
                               decoder_input,
                               mask_encoder_input,
                               mask_decoder_input))
        # if count == 100000:
        #     break
        count += 1

    encoder_input, \
    decoder_input, \
    mask_encoder_input, \
    mask_decoder_input = zip(*train_data)

    encoder_input = torch.stack(encoder_input)
    decoder_input = torch.stack(decoder_input)
    mask_encoder_input = torch.stack(mask_encoder_input)
    mask_decoder_input = torch.stack(mask_decoder_input)

    train_data = [encoder_input, decoder_input, mask_encoder_input, mask_decoder_input]

    torch.save(train_data, file_name)


def get_splited_data_by_file(dataset_file):
    datasets = [[], [], []]

    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)

    for d in data[:]:
        lst = []
        dialogue_len = 0
        for x in d['Dialogue']:
            lst = x.split()
            dialogue_len += 1
            if len(lst) < 4:
                if dialogue_len == 2:
                    data.remove(d)
                    break
                # else:
                #     d['Dialogue'] = d['Dialogue'][:dialogue_len-2]

    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]

    # print(datasets)
    return datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_size', default=400, type=int, required=False)
    parser.add_argument('--decoder_size', default=100, type=int, required=False)

    parser.add_argument('--csv_datapath', default="/home1/deeksha/MedKGLM/src/Data/cdialog/csv_file/", type=str,
                        required=False)
    parser.add_argument('--json_datapath', default='/home1/deeksha/MedKGLM/src/Data/cdialog/json_files/',
                        type=str, required=False)
    parser.add_argument('--save', default='/home1/deeksha/MedKGLM/src/preprocessed_data/data_cdialog_rand/', type=str,
                        required=False)
    parser.add_argument('--merged_datapath',
                        default='/home1/deeksha/MedKGLM/src/Data/cdialog/merged/merged_file.json', type=str,
                        required=False)
    args = parser.parse_args()

    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
    # never_split=("[UNK]", "</s>", "[PAD]", "<s>", "[MASK]", "[END]"))
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",never_split=("<unk>", "</s>", "<pad>", "<s>", "<mask>", "</s>"))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True,
                                                 never_split=("<unk>", "</s>", "<pad>", "<s>", "<mask>", "</s>"))

    MAX_ENCODER_SIZE = args.encoder_size
    MAX_DECODER_SIZE = args.decoder_size

    tot = 0
    maxi = 0
    mini = 1e5

    path1 = args.csv_datapath
    # path2 = args.Icliniq_datapath
    path3 = args.json_datapath

    total = [[], [], []]

    for filename in os.listdir(path1):
        # print(filename)
        dataset_file = os.path.join(path1, filename)
        final_file = os.path.join(path3, (filename.split(".")[0] + ".json"))
        # print(dataset_file)
        nos = clean_dataset(dataset_file, final_file)

    # for filename in os.listdir(path2):
    #     dataset_file = os.path.join(path2, filename)
    #     final_file = os.path.join(path3, (filename.split(".")[0] + ".json"))
    #     # print(dataset_file)
    #     nos = clean_dataset(dataset_file, final_file)

    # for filename in os.listdir(path3):
    #     dataset_file = os.path.join(path3, filename)
    #     temp = get_splited_data_by_file(dataset_file)
    #     # print(dataset_file)
    #     total[0].extend(temp[0])
    #     total[1].extend(temp[1])
    #     total[2].extend(temp[2])

    result = []
    for filename in glob.glob("/home1/deeksha/MedKGLM/src/Data/cdialog/json_files/*.json"):
        with open(filename, "r") as infile:
            result.extend(json.load(infile))
            random.shuffle(result)

    json_file = args.merged_datapath

    with open(json_file, "w") as outfile:
        json.dump(result, outfile)

    temp = get_splited_data_by_file(json_file)
    total[0].extend(temp[0])
    total[1].extend(temp[1])
    total[2].extend(temp[2])

    data = total

    print('train: ', len(data[0]))
    print('validate: ', len(data[1]))
    print('test: ', len(data[2]))

    ntr = 0
    nv = 0
    nte = 0

    for x in data[0]:
        ntr += len(x)

    for y in data[1]:
        nv += len(y)

    for z in data[2]:
        nte += len(z)

    print('train # ut: ', ntr)
    print('validate # ut: ', nv)
    print('test # ut: ', nte)

    # print(f'Process the train dataset')
    # make_dataset(data[0], args.save + '/train_data.pkl')
    #
    # print(f'Process the validate dataset')
    # make_dataset(data[1], args.save + '/validate_data.pkl')
    #
    # print(f'Process the test dataset')
    # make_dataset(data[2], args.save + '/test_data.pkl')
    #
    # print("#############")
    # print(tot)
    # print(maxi)
    # print(mini)

    '''
    clean_dataset("/home1/deeksha/CDialog/src/Data/MedDialogCorpus/HCM/healthcaremagic_dialogue_1.txt",
                  "/home1/deeksha/CDialog/src/Data/MedDialogCorpus/json_files_1/healthcaremagic_dialogue_1.json" )
    '''
