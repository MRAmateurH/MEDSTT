# Built-in Packages
import os
import sys
from urllib.parse import urlparse
import json
from argparse import ArgumentParser
import time
import datetime
import unicodedata

# 
from lib.Spider import *
from lib.Cloud import TxtToSpeech, SpeechToText, Test
# from lib.log import log
import lib.log as log

# External Packages
import html
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import openai
import jiwer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import spacy

nlp = spacy.load("en_core_web_sm")


def start() -> datetime:
    start_time = datetime.datetime.now()
    print(f'開始時間： {start_time}')
    return start_time

def end(start_time) -> None:
    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    print(f'開始時間： {start_time}')
    print(f'結束時間： {end_time}')
    print(f"時間差： {time_diff}")

    total_minutes = time_diff.total_seconds() / 60  # 計算總共的分鐘數
    print(f"總共分鐘數：{int(total_minutes)}")

# --option: crawl
def read_txt_file(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            yield line.strip()

def extract_urls(lines):
    for line in lines:
        url_parts = urlparse(line)
        if url_parts.scheme == 'http' or url_parts.scheme == 'https':
            yield url_parts.geturl()

def segment_sentences(text: str) -> list:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def group_sentences(text: str) -> list:
    results = []
    if len(text) > 5000:
        sents = segment_sentences(text)
        start=0
        while start < len(sents):
            end = start
            count = 0
            while count < 5000 and end < len(sents):
                count += len(sents[end])
                end += 1
            # 如果第5000個字不能包含一個完整的句子，則只讀到n-1個句子
            if count > 5000:
                end -= 1
            results.append(''.join(sents[start:end]))
            start = end
    else:
        results.append(text)

    return results

def toJson(dialogueObject: dict, file='./json_output/sample.json'):
    dialogueArray = []

    if os.path.exists(file):
        with open(file, 'r') as inputfile:
            data = json.load(inputfile)
        data.append(dialogueObject)
    else:
        data = dialogueArray
    dialogueArray.append(dialogueObject)

    with open(file, 'w') as outputfile:
        outputfile.write(json.dumps(data, indent=4))

# --option: wer
def add_result(audio_output_dir: str, jsonfile: str, outfile: str) -> None:
    '''
    Adds the results from audio output files to the corresponding entries in a JSON file.

    Args:
        audio_output_dir (str): The directory path containing the audio output files.
        jsonfile (str): The path to the JSON file.
        outfile (str): The path to the output JSON file with added results.

    Returns:
        None

    Example Usage:
        >>> add_result('audio_output/', 'data.json', 'data_with_results.json')
    '''
    # Load the JSON file
    with open(jsonfile, 'r') as f:
        data = json.load(f)

    # Initialize the 'results' dictionary for each entry in the JSON data
    for i in data:
        i['results'] = dict()

    # Iterate through the files in the audio output directory
    
    for _, file in tqdm(enumerate(os.listdir(audio_output_dir)), desc='Add result', ncols=75):
        if 'output' in file:  # Process only files with 'output' in their name
            split = file.split('_')
            file_row = split[1]
            file_id = int(split[2][:-4])

            # Read the content from the audio output file
            with open(f'{audio_output_dir}{file}', 'r') as input:
                content = input.read()
                # Update the corresponding entry in the JSON data with the content
                data[file_id-1]['results'][file_row] = content

    # Write the updated JSON data to the output file
    with open(outfile, 'w') as out:
        json.dump(data, out)

    with open(outfile, 'w') as output:
        output.write(json.dumps(data, indent=4))

def preprocess(text: str) -> str:
    '''
    Preprocesses the given text by applying tokenization, lowercase conversion, and punctuation removal.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.

    Example Usage:
        >>> preprocessed_text = preprocess("Hello, World!")
        >>> print(preprocessed_text)
        "hello world"
    '''
    # Tokenize the text and apply preprocessing steps
    doc = nlp(text)

    # Join the lowercase text of non-punctuation tokens into a single string
    preprocessed_text = ' '.join([token.text.lower().strip() for token in doc if not token.is_punct])

    return preprocessed_text

def segment(text: str) -> list:
    '''
    '''
    doc = nlp(text)

    preprocessed_text = [token.text.lower().strip() for token in doc if not token.is_punct]

    return preprocessed_text


def calculate_wer(reference: str, hypothesis: str) -> float:
    '''
    Calculates the Word Error Rate (WER) between a reference and a hypothesis.

    Args:
        reference (str): The reference string.
        hypothesis (str): The hypothesis string.

    Returns:
        float: The WER value as a percentage.

    Example Usage:
        >>> wer = calculate_wer("hello world", "hello there")
        >>> print(wer)
        40.00
    '''
    return jiwer.wer(preprocess(reference), preprocess(hypothesis)) * 100

def calculate_average_wer(wer_list: list) -> str:
    '''
    Calculates the average Word Error Rate (WER) from a list of WER values.

    Args:
        wer_list (list): A list of WER values.

    Returns:
        str: The average WER value as a formatted string with two decimal places followed by "%",
             or 'N/A' if the list is empty.

    Example Usage:
        >>> average_wer = calculate_average_wer([10.5, 20.0, 15.25])
        >>> print(average_wer)
        '15.25%'
    '''
    # for idx, i in enumerate(wer_list, start=1):
    #     print(idx, i)
    if wer_list:
        average_wer = '{:.2f}%'.format(sum(wer_list) / len(wer_list))
    else:
        average_wer = 'N/A'
    return average_wer

def wer(jsonfile: str, ref: str='utterances', hyp: str='results') -> dict:
    '''
    Calculates the Word Error Rate (WER) for a given JSON file.

    Args:
        jsonfile (str): The path to the JSON file containing utterances and results.

    Returns:
        dict: A dictionary containing WER values for each utterance key.

    Example Usage:
        >>> result = wer('data.json')
        >>> print(result)
        {'utterance1_wer_list': [0.25, 0.5, 0.75], 'utterance2_wer_list': [0.1, 0.2, 0.3]}
    '''
    with open(jsonfile, 'r') as input:
        contents = json.load(input)

    if not contents:
        return {}

    reference_key = contents[0][ref].keys()
    result_dict = dict()

    for key in reference_key:
        result_dict[f'{key}_wer_list'] = []

    for content in tqdm(contents, desc='Calculating WER', ncols=75):
        for key in reference_key:
            reference = content[ref][key]
            hypothesis = content[hyp][key]

            wer = calculate_wer(reference, hypothesis)

            result_dict[f'{key}_wer_list'].append(wer)

    return result_dict

def calculate_average_bleu(bleu_list: list):
    '''
    '''
    if bleu_list:
        average_bleu = '{:.2f}%'.format(sum(bleu_list) / len(bleu_list))
    else:
        average_bleu = 'N/A'
    return average_bleu

def bleu(jsonfile: str, ref: str='utterances', hyp: str='results') -> dict:
    '''
    '''
    with open(jsonfile, 'r') as input:
        contents = json.load(input)

    if not contents:
        return {}

    reference_key = contents[0][ref].keys()
    result_dict = dict()

    for key in reference_key:
        result_dict[f'{key}_bleu_score'] = []

    for content in tqdm(contents, desc='Calculating BLEU Score', ncols=90):
        for key in reference_key:
            reference = [segment(content[ref][key])]
            hypothesis = segment(content[hyp][key])

            score = sentence_bleu(reference, hypothesis)

            result_dict[f'{key}_bleu_score'].append(score)

    return result_dict


# --option: grammar correction
def grammar_correction(sentence: str) -> str:
    '''
    Corrects grammar and formatting issues in a given sentence using OpenAI's text-davinci-003 model.

    Args:
        sentence (str): The input sentence to be corrected.

    Returns:
        str: The corrected sentence.

    Example Usage:
        >>> corrected_sentence = grammar_correction("Ths is an example sentence.")
        >>> print(corrected_sentence)
        "This is an example sentence."
    '''
    # Set the OpenAI API key from the API key file
    openai.api_key_path = './auth/api_key.txt'
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Set up the prompt for the grammar correction request
    prompt = "Please correct the sentence by removing special symbols, \
                replacing numbers with ordinal words (e.g., 'First', 'Second'), \
                and express all symbols using textual representation. \
                If there are any email-like, phone-like, or address-like strings, delete the sentences containing those strings. \
                Delete any information related to 'Dr.' (name, phone, etc.). \
                Also, ensure that each word at the start of a sentence is preceded by a space.\n"


    prompt += f'sentence: "{sentence}"'
    max_tokens = int(len(prompt.split())*1.5)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='actions', title='actions', help="sub-command: help")

    # Subparser for option 'crawl'
    parser_crawl = subparsers.add_parser('crawl', help='option crawl')
    parser_crawl.add_argument('-i', '--input', help='choose an input file', required=True)
    parser_crawl.add_argument('-o', '--output', help="Enter an output file name. If nothing is entered, the default output name will be 'sample'.")

    # Subparser for option 'tts'
    parser_tts = subparsers.add_parser('texttospeech', help='option text to speech')
    parser_tts.add_argument('-i', '--input', help='choose an input file', required=True)
    parser_tts.add_argument('-o', '--output', help='choose an output directory', required=True)

    # Subparser for option 'add_result'
    parser_add_result = subparsers.add_parser('add_result', help='option add result')
    parser_add_result.add_argument('-d', '--directory', help='choose an aspire speech to text output file directory', required=False)
    parser_add_result.add_argument('-i', '--input', help='choose an testing dataset (JSON)', required=False)
    parser_add_result.add_argument('-o', '--output', help='choose or create an output file (JSON)', required=False)

    # Subparser for option 'wer'
    parser_wer = subparsers.add_parser('wer', help='option wer')
    parser_wer.add_argument('-i', '--input', help='choose an testing dataset (JSON)', required=True)
    parser_wer.add_argument('--ref', help='the key of the reference in the input json file', required=True)
    parser_wer.add_argument('--hyp', help='the key of the hypothesis in the input json file', required=True)

    # Subparser for option 'bleu'
    parser_bleu = subparsers.add_parser('bleu', help='option bleu')
    parser_bleu.add_argument('-i', '--input', help='choose an testing dataset (JSON)', required=True)
    parser_bleu.add_argument('--ref', help='the key of the reference in the input json file', required=True)
    parser_bleu.add_argument('--hyp', help='the key of the hypothesis in the input json file', required=True)

    # Subparser for option 'grammar correction'
    parser_grammar_correction = subparsers.add_parser('grammar_correction', help='option grammer correction')
    parser_grammar_correction.add_argument('-i', '--input', help='choose an input json file', required=True)
    parser_grammar_correction.add_argument('-o', '--output', help='choose an output file', required=True)

    parser_stt = subparsers.add_parser('speechtotext', help='option speech to text')
    parser_stt.add_argument('-i', '--input', help='choose an input audio directory', required=True)
    parser_stt.add_argument('-o', '--output', help='choose an output directory', required=True)

    args = parser.parse_args()
    actions = args.actions

    match actions:
        case 'crawl':
            start_time = start()
            
            try:
                # Get the argument value
                input_file = args.input
                output_file = args.output

                if os.path.exists(output_file):
                    with open(output_file, 'r') as file:
                        data = json.load(file)
                        last_id = data[-1]['id']
                        id = last_id+1
                else:
                    id = 1
                    
                if os.path.exists(input_file):
                    lines = read_txt_file(input_file)
                    for count, url in enumerate(extract_urls(lines)):
                        utterances, dialogueObject = dict(), dict()
                        print(f"It's now processing data {id}, and the URL is {url} ...")

                        req = requests.get(url)
                        soup = BeautifulSoup(req.content, 'html.parser')

                        description = Spider()
                        description.content = soup.find(class_="subheading text-primary")
                        description.remove_html_tag(description.content)
                        description.remove_parenthesis_content(description.content)
                        description.remove_extra_spaces(description.content)

                        patient = Spider()
                        patient.content = soup.find(class_="paragraph")
                        patient.remove_html_tag(patient.content)
                        patient.remove_parenthesis_content(patient.content)
                        patient.remove_extra_spaces(patient.content)

                        doctor = Spider()
                        doctor.content = soup.find(class_="card plantfood")
                        doctor.remove_html_tag(doctor.content)
                        doctor.remove_parenthesis_content(doctor.content)
                        doctor.remove_extra_spaces(doctor.content)
                        doctor.remove_url(doctor.content)
                        doctor.normalize_quotes(doctor.content)

                        utterances['patient'] = patient.content
                        utterances['doctor'] = doctor.content

                        dialogueObject['id'] = id
                        dialogueObject['description'] = description.content
                        dialogueObject['utterances'] = utterances
                        
                        # save to json
                        if output_file is not None:
                            toJson(dialogueObject, file=output_file)
                        else:
                            toJson(dialogueObject)
                        id += 1
            except:
                pass
            end(start_time)
            print(f'總共寫入 {count} 篇文章')

        case 'texttospeech':
            '''
            '''
            # Get the argument value
            input_file = args.input
            output_dir = args.output

            start_time = start()
            id = 0
            try:
                tts = TxtToSpeech(credentials='./auth/inspired-fact-383607-aa572fa1b7fa.json')
                with open(input_file, 'r') as input:
                    data = json.load(input)

                for idx, text in enumerate(data, start=1):
                    pubmed = text['utterances']['pubmed']
                    sents = group_sentences(pubmed)

                    for sent in sents:
                        filename_split = os.path.normpath(output_dir).split(os.sep)
                        filename = filename_split[-1]
                        output_file = os.path.join(output_dir, f'{filename}_{idx}.wav')
                        tts.txttospeech(sent, output_file)
                    id+=1

                # for idx, text in enumerate(data, start=1):
                #     sents = group_sentences(text)
                #     for sent in sents:
                #         filename = output_dir.split('/')[-1]
                #         output_file = os.path.join(output_dir, f'{filename}_{idx}.wav')
                #         tts.txttospeech(output_dir, sent, output_file)
                #     id+=1
                
                # for idx, dialogue in enumerate(data, start=1):
                #     patient = dialogue['utterances']['patient']
                #     doctor = dialogue['utterances']['doctor']
                    
                #     patient_sents = group_sentences(patient)
                #     for sent in patient_sents:
                #         tts.txttospeech(output_dir, idx, 'patient', sent)
                #         # print(f'{idx}\tpatient:')
                #         # print(len(patient_sents), patient_sents, '\n')
                    
                
                #     doctor_sents = group_sentences(doctor)
                #     for sent in doctor_sents:
                #         tts.txttospeech(output_dir, idx, 'doctor', sent)
                #         # print(f'{idx}\tdoctor:')
                #         # print(len(doctor_sents), doctor_sents, '\n')

            except Exception as e:
                print(e)
            
            end(start_time)
            print(f'總共產生 {id} 個音檔')

        case 'add_result':
            directory = args.directory
            input_file = args.input
            output_file = args.output  
            
            # add_result('./output_8000_test1/', './sample.json', './results1.json')
            add_result(audio_output_dir=directory, jsonfile=input_file, outfile=output_file)
        
        case 'wer':
            # Get the argument value
            input_file = args.input
            ref = args.ref
            hyp = args.hyp
            
            wer_dict = wer(input_file, ref=ref, hyp=hyp)

            for key in wer_dict.keys():
                role = key.split('_')[0]
                average_wer = calculate_average_wer(wer_dict[key])
                print(f'{role} word error rate: {average_wer}')

        case 'bleu':
            # Get the argument value
            input_file = args.input
            ref = args.ref
            hyp = args.hyp
            
            bleu_dict = bleu(input_file, ref=ref, hyp=hyp)

            for key in bleu_dict.keys():
                role = key.split('_')[0]
                average_bleu = calculate_average_bleu(bleu_dict[key])
                print(f'{role} avg. BLEU score: {average_bleu}')

        case 'grammar_correction':
            # Get the argument value
            input_file = args.input
            openai_generate_file = args.output

            start_time = start()

            if os.path.exists(openai_generate_file):
                with open(openai_generate_file, 'r') as file:
                    data = json.load(file)
                    last_id = data[-1]['id']
                    id = last_id+1
            else:
                id = 1
                
            try:
                with open(input_file, 'r') as file:
                    data = json.load(file)

                for idx, dialogue in enumerate(data, start=1):
                    patient_text = dialogue['utterances']['patient']
                    doctor_text = dialogue['utterances']['doctor']
                    # 將全形轉為半形
                    normalized_patient_text = unicodedata.normalize('NFKC', patient_text)
                    normalized_doctor_text = unicodedata.normalize('NFKC', doctor_text)
                    # HTML編碼轉換 &amp; -> '&'; &gt; -> '>'
                    normalized_patient_text = html.unescape(normalized_patient_text)
                    normalized_doctor_text = html.unescape(normalized_doctor_text)

                    patient_response = grammar_correction(normalized_patient_text)['choices'][0]['text'].replace('\n', ' ')
                    doctor_response = grammar_correction(normalized_doctor_text)['choices'][0]['text'].replace('\n', ' ')

                    if patient_response == '' or doctor_response == '':
                        print(f'{idx}th data don\'t have response')

                    dialogue['id'] = id
                    dialogue['utterances']['patient'] = patient_response
                    dialogue['utterances']['doctor'] = doctor_response

                    toJson(dialogue, file=openai_generate_file)
                    id += 1
            except Exception as e:
                print(e)

            end(start_time)
            print(f'總共寫入 {idx} 篇文章')
        
        case 'speechtotext':
            input_directory = args.input
            output_directory = args.output

            start_time = start()
            os.makedirs(output_directory, exist_ok=True)

            try:
                speech = SpeechToText(credentials='./auth/inspired-fact-383607-aa572fa1b7fa.json')
                for audio_file in os.listdir(input_directory):
                    audio_split = audio_file.split('_')
                    role = audio_split[0]
                    dataset_idx = audio_split[1].split('.')[0]
                    
                    response_text = speech.SpeechToText(speech_file=os.path.join(input_directory, audio_file))

                    output_file = os.path.join(output_directory, f'output_{role}_{dataset_idx}.txt')
                    with open(output_file, 'w') as out:
                        out.write(response_text)
            except Exception as e:
                print(e)

if __name__ == '__main__':
    # log_obj = log()
    # log_obj.init()
    
    # test = Test()
    # test.test_log()
    # log.init()

    main()
    