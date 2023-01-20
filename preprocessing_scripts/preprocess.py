from transformers import AutoTokenizer
import torch
import os
import codecs
import json
import glob
import random
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

def clean_dataset(raw, final_file):
    f_in = open(raw, "r")
    f_wr = open(final_file, "w", encoding='utf-8')

    # i=0
    Dialogue_list = []
    Total=0

    while True:
        line = f_in.readline()
        # print(line)
        if not line:
            break
        # i+=1
        
        if line[:11] == "Description":
            # print("back")
            first_list=[]
            first_turn = 0
            first_dialog = {}

            desp_l= "Patient: "
            # print(line)
            while True:
                desp_start = f_in.readline()
                # print(desp_start)

                desp_utterance = desp_start.rstrip()
                if desp_utterance == "":
                    continue

                if desp_utterance:
                    if desp_utterance[-1] not in ".。？，,?!！~～;":
                        desp_utterance += '。'

                if desp_utterance:
                    desp_l =desp_l + desp_utterance

                break

            first_utterance = desp_l
            
            # print(first_utterance)

        elif line[:8] == "Dialogue":
            first_turn = 1
            while True:

                dia_start = f_in.readline()
                # print("dia", dia_start)

                if (not dia_start) or (dia_start in ["\n", "\n\r"]):
                    # print("broke")
                    if int(first_turn/2)>0:
                        temp = int(first_turn/2)
                        first_dialog["Turn"] = temp
                        Total+= 1
                        first_dialog["Id"] = Total
                        first_dialog["Dialogue"] = first_list[:(2*temp)]
                        Dialogue_list.append(first_dialog)

                    break

                # print(dia_start)
                if dia_start[:8] == "Patient:":
                    pat_utt=""
                    while True:

                        patient_st = f_in.readline()
                        # print(patient_st)

                        pat_utterance = patient_st.rstrip()

                        # print(pat_utterance)

                        if pat_utterance != "":
                            if pat_utterance != "Doctor:":
                                if pat_utterance[-1] not in ".。？，,?!！~～;":
                                    pat_utterance += '。'

                        else:
                            break

                        # print("r",pat_utterance)
                        if pat_utterance != "Doctor:":
                            # print("x", pat_utterance)
                            pat_utt  = pat_utt+ pat_utterance
                            # print("y",pat_utt)
                            file_pointer = f_in.tell()
                        else:
                            f_in.seek(file_pointer)
                            break
                        # print("ff", first_utterance)
                        # print("\n")
                        # print("pp", pat_utt)
                        # print("\n")
                        if match(first_utterance, pat_utt):
                            pad_f_utt= first_utterance
                        else:
                            pad_f_utt= first_utterance + pat_utt
                    if pad_f_utt != "":
                        first_list.append(pad_f_utt)

                    if pat_utterance == "":
                        continue
                    # print(pad_f_utt)
                    # print("\n")

                elif dia_start[:7] == "Doctor:":
                    first_turn+=1
                    doc_utt="Doctor:"
                    while True:
                        
                        doctor_st = f_in.readline()

                        doc_utterance = doctor_st.rstrip()

                        if doc_utterance != "":
                            if doc_utterance[-1] not in ".。？，,?!！~～;":
                                doc_utterance += '。'
                        else:
                            break

                        if doc_utterance:

                            doc_utt  = doc_utt+ doc_utterance

                            file_pointer = f_in.tell()
                        else:
                            f_in.seek(file_pointer)
                            break

                    if doc_utt != "":
                        first_list.append(doc_utt)

                    if doc_utterance == "":
                        continue
                    # print(doc_utt)
                    # print("\n")

            # print(first_dialog)   # here you can print dialogues and see
            # print("\n")
            # print(final_file)

    print("Total Cases: ", final_file.split('.')[0].split('/')[-1], Total)
    json.dump(Dialogue_list, f_wr, ensure_ascii=False, indent=4)
    f_in.close()
    f_wr.close()
    return Total

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
    encoder_input = tokenizer((' ').join(encoder_input))['input_ids'][1:-1][:MAX_ENCODER_SIZE-1] + [tokenizer((' ').join(encoder_input))['input_ids'][-1]] #tokenizer.convert_tokens_to_ids(encoder_input) 
    decoder_input = tokenizer((' ').join(decoder_input))['input_ids'][1:-1][:MAX_DECODER_SIZE-1] + [tokenizer((' ').join(decoder_input))['input_ids'][-1]] #tokenizer.convert_tokens_to_ids(decoder_input) 

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
    parser.add_argument('--HCM_datapath', default="/home1/deeksha/MedKGLM/src/Data/MedDialogCorpus/HCM/", type=str, required=False)
    parser.add_argument('--Icliniq_datapath', default='/home1/deeksha/MedKGLM/src/Data/MedDialogCorpus/Icliniq/', type=str, required=False)
    parser.add_argument('--json_datapath', default='/home1/deeksha/MedKGLM/src/Data/MedDialogCorpus/json_files/', type=str, required=False)
    parser.add_argument('--save', default='/home1/deeksha/MedKGLM/src/preprocessed_data/data_med/', type=str, required=False)
    args = parser.parse_args()

    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', \
                                          # never_split=("[UNK]", "</s>", "[PAD]", "<s>", "[MASK]", "[END]"))
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base",never_split=("<unk>", "</s>", "<pad>", "<s>", "<mask>", "</s>"))
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", add_prefix_space=True,never_split=("<unk>", "</s>", "<pad>", "<s>", "<mask>", "</s>"))
    


    MAX_ENCODER_SIZE = args.encoder_size
    MAX_DECODER_SIZE = args.decoder_size

    tot = 0
    maxi = 0
    mini = 1e5

    path1 = args.HCM_datapath
    path2 = args.Icliniq_datapath
    path3 = args.json_datapath

    total = [[],[],[]]


    for filename in os.listdir(path1):
        # print(filename)
        dataset_file = os.path.join(path1, filename)
        final_file = os.path.join(path3, (filename.split(".")[0]+".json"))
        # print(dataset_file)
        nos = clean_dataset(dataset_file, final_file)


    for filename in os.listdir(path2):
        dataset_file = os.path.join(path2, filename)
        final_file = os.path.join(path3, (filename.split(".")[0]+".json"))
        # print(dataset_file)
        nos = clean_dataset(dataset_file, final_file)

    
    for filename in os.listdir(path3):
        dataset_file = os.path.join(path3, filename)
        temp= get_splited_data_by_file(dataset_file)
        # print(dataset_file)
        total[0].extend(temp[0])
        total[1].extend(temp[1])
        total[2].extend(temp[2])

    data = total

    print(len(data[0]))
    print(len(data[1]))
    print(len(data[2]))

    print(f'Process the train dataset')
    make_dataset(data[0], args.save + '/train_data.pkl')

    print(f'Process the validate dataset')
    make_dataset(data[1], args.save + '/validate_data.pkl')

    print(f'Process the test dataset')
    make_dataset(data[2], args.save + '/test_data.pkl')

    print("#############")
    print(tot)
    print(maxi)
    print(mini)


    '''
    clean_dataset("/home1/deeksha/CDialog/src/Data/MedDialogCorpus/HCM/healthcaremagic_dialogue_1.txt",
                  "/home1/deeksha/CDialog/src/Data/MedDialogCorpus/json_files_1/healthcaremagic_dialogue_1.json" )
    '''