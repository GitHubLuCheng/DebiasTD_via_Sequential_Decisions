import pickle
import os
import string
import csv
import json
from googleapiclient import discovery

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
from nltk.corpus import stopwords
from aae import predict
import time
from tqdm import tqdm


def collate_fn(batch):
    batch = filter(lambda x: x is not None, batch)
    docs, labels, doc_lengths, sent_lengths, groups_1, groups_2 = list(zip(*batch))

    bsz = len(labels)
    batch_max_doc_length = max(doc_lengths)
    batch_max_sent_length = max([max(sl) if sl else 0 for sl in sent_lengths])

    docs_tensor = torch.zeros((bsz, batch_max_doc_length, batch_max_sent_length)).long()
    sent_lengths_tensor = torch.zeros((bsz, batch_max_doc_length)).long()

    for doc_idx, doc in enumerate(docs):
        doc_length = doc_lengths[doc_idx]
        sent_lengths_tensor[doc_idx, :doc_length] = torch.LongTensor(sent_lengths[doc_idx])
        for sent_idx, sent in enumerate(doc):
            sent_length = sent_lengths[doc_idx][sent_idx]
            docs_tensor[doc_idx, sent_idx, :sent_length] = torch.LongTensor(sent)

    return docs_tensor, torch.LongTensor(labels), torch.LongTensor(doc_lengths), sent_lengths_tensor, torch.LongTensor(groups_1), torch.LongTensor(groups_2)
    
##### ADDED INSTAGRAM DATASET #####
# create the dataset
class InstagramDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, remove_puncs=True, comments_aae=False, pre_loaded_data=None, max_comments=None, window_size=None, offset=None):
        MAX_SENT_LENGTH = 150
        # MAX_SENTS = 40 if max_comments == None else max_comments
        self.max_comments = max_comments
        MAX_SENTS = 1000
        self.window_size = window_size
        self.offset = offset

        if window_size != None and offset == None:
            raise ValueError

        if pre_loaded_data != None:
            # load from array!
            self.labels, self.data = pre_loaded_data[0], pre_loaded_data[1]

            self.vocab = pd.read_csv(
                filepath_or_buffer='data/glove/glove.6B.100d.txt',
                header=None,
                sep=" ",
                quoting=csv.QUOTE_NONE,
                usecols=[0]).values[:50000]
            self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
            self.comments_aae = comments_aae

            if comments_aae:
                self.aae_model = predict.load_model()
            else:
                self.aae_model = None

            puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

            return 

        data = pd.read_csv(datapath, sep='\t', names=["sentiment", "review"])

        self.vocab = pd.read_csv(
            filepath_or_buffer='data/glove/glove.6B.100d.txt',
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
        self.comments_aae = comments_aae

        if comments_aae:
            self.aae_model = predict.load_model()
        else:
            self.aae_model = None

        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        # split into sentences and words
        if not os.path.exists('./data/insta_cache.pkl'):
            self.data = []
            self.labels = []

            for idx in tqdm(range(data.review.shape[0])):
                text = self.preprocess_text(data.review[idx])

                doc = [
                    [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
                    for sent in sent_tokenize(text=text) if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
                doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
                num_sents = min(len(doc), MAX_SENTS)

                if num_sents == 0:
                    continue

                num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

                self.labels.append(data.sentiment[idx])
                self.data.append((
                    doc,
                    num_sents,
                    num_words
                ))
            pickle.dump([self.labels, self.data], open('./data/insta_cache.pkl', 'wb'))
        else:
            print('Loading Data From Cache :)')
            self.labels, self.data = pickle.load(open('./data/insta_cache.pkl', 'rb'))

    def preprocess_text(self, text):
        for c in ['\'', '"', '@', ',', '#', '%%', '.', '!']:
            text = text.replace(c, '')
        return text.strip().lower()

    def transform(self, text, remove_puncs=True):
        MAX_SENT_LENGTH = 150
        MAX_SENTS = 40 if self.max_comments == None else self.max_comments
        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        doc = [
            [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
            for sent in sent_tokenize(text=text) if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
        doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
        num_sents = min(len(doc), MAX_SENTS)

        if num_sents == 0:
            return None, 0, None

        num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

        return doc, num_sents, num_words, -1

    def get_perspective_comment_labels(self):
        if not os.path.exists('insta_pers_cache.pkl'):
            API_KEY = 'AIzaSyAv6aH0IbcCweqJjM3JoV_M3vqszQdmAa0'

            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            self.comments_pers_labels = []
            # set labels according to the keywords
            for data, _, _ in tqdm(self.data):
                # loop on sentences
                session_comments_labels = []
                for sent in data:
                    start_time = time.time()
                    # loop on sentences (comments?)
                    comment = ' '.join(list(map(lambda f: self.vocab[f], sent)))
                    
                    # get the comment label
                    analyze_request = {
                        'comment': { 'text': comment },
                        'requestedAttributes': {'TOXICITY': {}},
                        'languages': ['en']
                    }
                    response = client.comments().analyze(body=analyze_request).execute()

                    session_comments_labels.append( 
                        response['attributeScores']['TOXICITY']['summaryScore']['value']
                    )

                    end_time = time.time()
                    time.sleep(1.1 - (start_time - end_time))

                # add to self
                self.comments_pers_labels.append(session_comments_labels)
            pickle.dump(self.comments_pers_labels, open('insta_pers_cache.pkl', 'wb'))
        else:
            print('Pers results from cache!')
            self.comments_pers_labels = pickle.load(open('insta_pers_cache.pkl', 'rb'))

    def create_comment_labels(self, keywords_file='./bad-keywords.txt'):
        # use keyword's file to label each comment
        self.comments_labels = []
        self.comments_labels_dialect = []
        self.group = []
        self.group_dialect = []

        bad_keywords = []
        with open(keywords_file, 'r') as fin:
            for line in fin.readlines():
                bad_keywords.append(line.strip().lower())
            fin.close()

        # set labels according to the keywords
        for data, _, _ in self.data:
            # loop on sentences
            comments_labels = []
            comments_labels_dialect = []
            for sent in data:
                # loop on sentences (comments?)
                comment = ' '.join(list(map(lambda f: self.vocab[f], sent)))
                labeled = False

                for b in bad_keywords:
                    if b in comment:
                        comments_labels.append(1)
                        labeled = True
                        break
                if not labeled:
                    comments_labels.append(0)
                
                comments_labels_dialect.append( np.argmax(predict.predict(comment.split())) )

            # add to self
            self.comments_labels.append(comments_labels)
            self.comments_labels_dialect.append(comments_labels_dialect)

            # set the data group
            group = 1 if sum(comments_labels) > len(comments_labels) / 2 else 0
            self.group.append(group)

            freq = {x:comments_labels_dialect.count(x) for x in comments_labels_dialect}
            self.group_dialect.append(max(freq, key=freq.get))

    def __getitem__(self, idx, comments_details=False):
        if self.window_size == None or self.offset + self.window_size >= len(self.data[idx][0]):
            max_comments = min(len(self.data[idx][0]), self.max_comments)
            comments = self.data[idx][0][:max_comments]
            num_sents = min(len(comments), max_comments)
            num_words = [min(len(sent), 150) for sent in comments][:max_comments]
        elif self.offset + self.window_size < len(self.data[idx][0]):
            max_comments = min(len(self.data[idx][0]), self.window_size)
            comments = self.data[idx][0][self.offset : self.offset + self.window_size]
            num_sents = min(len(comments), max_comments)
            num_words = [min(len(sent), 150) for sent in comments][:max_comments]
    
        if not comments_details:
            return comments, self.labels[idx], num_sents, num_words, self.group[idx], self.group_dialect[idx]
        return comments, self.labels[idx], num_sents, num_words, self.comments_labels[idx], self.comments_labels_dialect[idx]
        
    def __len__(self):
        return len(self.labels)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def num_classes(self):
        return 2



###### VINE DATASET ######
class VineDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, labelspath, remove_puncs=True, comments_aae=False, pre_loaded_data=None, max_comments=None):
        MAX_SENT_LENGTH = 150
        # MAX_SENTS = 40 if max_comments == None else max_comments
        self.max_comments = max_comments
        MAX_SENTS = 1000

        if pre_loaded_data != None:
            # load from array!
            self.labels, self.data = pre_loaded_data[0], pre_loaded_data[1]
            self.vocab = pd.read_csv(
                filepath_or_buffer='data/glove/glove.6B.100d.txt',
                header=None,
                sep=" ",
                quoting=csv.QUOTE_NONE,
                usecols=[0]).values[:50000]
            self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
            self.comments_aae = comments_aae

            if comments_aae:
                self.aae_model = predict.load_model()
            else:
                self.aae_model = None

            puncs_table = str.maketrans(dict.fromkeys(string.punctuation))
            return

        # load the data here, list of sessions
        data = []
        labels = []
        with open(datapath, 'r') as json_file:
            json_list = list(json_file)
            json_file.close()
        for json_str in json_list:
            data.append(json.loads(json_str))
        
        if comments_aae:
            self.aae_model = predict.load_model()
        else:
            self.aae_model = None

        # read the labels here
        with open(labelspath, 'r') as cls_file:
            for line in cls_file.readlines():
                if line[0] == '$':
                    continue
                _, label = list(map(int, line.strip().split(' ')))
                labels.append(label)

            cls_file.close()
        
        print('Data Stats: %d | Labels: %d' %(len(data), len(labels)))

        self.vocab = pd.read_csv(
            filepath_or_buffer='data/glove/glove.6B.100d.txt',
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        # split into sentences and words
        if not os.path.exists('./data/vine_cache.pkl'):
            self.data = []
            self.labels = []

            for idx in tqdm(range(len(data))):
                comments = [c[1] for c in data[idx]['comments_list']]

                doc = [
                    [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
                    for sent in comments if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
                doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
                num_sents = min(len(doc), MAX_SENTS)

                if num_sents == 0:
                    continue

                num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

                self.labels.append(labels[idx])
                self.data.append((
                    doc,
                    num_sents,
                    num_words
                ))
            pickle.dump([self.labels, self.data], open('./data/vine_cache.pkl', 'wb'))
        else:
            print('Loading Data From Cache :)')
            self.labels, self.data = pickle.load(open('./data/vine_cache.pkl', 'rb'))

    def preprocess_text(self, text):
        for c in ['\'', '"', '@', ',', '#', '%%', '.', '!']:
            text = text.replace(c, '')
        return text.strip().lower()

    def transform(self, comments_list, remove_puncs=True):
        MAX_SENT_LENGTH = 150
        MAX_SENTS = 40 if self.max_comments == None else self.max_comments
        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        doc = [
            [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
            for sent in comments_list if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
        doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
        num_sents = min(len(doc), MAX_SENTS)

        if num_sents == 0:
            return None, 0, None

        num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

        return doc, num_sents, num_words, -1

    def get_perspective_comment_labels(self):
        if not os.path.exists('vine_pers_cache.pkl'):
            API_KEY = 'AIzaSyAv6aH0IbcCweqJjM3JoV_M3vqszQdmAa0'

            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            self.comments_pers_labels = []

            # set labels according to the keywords
            for data, _, _ in tqdm(self.data):
                # loop on sentences
                session_comments_labels = []
                for sent in data:
                    start_time = time.time()
                    # loop on sentences (comments?)
                    comment = ' '.join(list(map(lambda f: self.vocab[f], sent)))
                    
                    # get the comment label
                    analyze_request = {
                        'comment': { 'text': comment },
                        'requestedAttributes': {'TOXICITY': {}},
                        'languages': ['en']
                    }
                    response = client.comments().analyze(body=analyze_request).execute()

                    session_comments_labels.append( 
                        response['attributeScores']['TOXICITY']['summaryScore']['value']
                    )

                    end_time = time.time()
                    time.sleep(1.1 - (start_time - end_time))

                # add to self
                self.comments_pers_labels.append(session_comments_labels)
            pickle.dump(self.comments_pers_labels, open('vine_pers_cache.pkl', 'wb'))
        else:
            print('Pers results from cache!')
            self.comments_pers_labels = pickle.load(open('vine_pers_cache.pkl', 'rb'))

    def create_comment_labels(self, keywords_file='./bad-keywords.txt'):
        # use keyword's file to label each comment
        self.comments_labels = []
        self.comments_labels_dialect = []
        self.group = []
        self.group_dialect = []

        bad_keywords = []
        with open(keywords_file, 'r') as fin:
            for line in fin.readlines():
                bad_keywords.append(line.strip().lower())
            fin.close()

        # set labels according to the keywords
        for data, _, _ in self.data:
            # loop on sentences
            comments_labels, comments_labels_dialect = [], []
            for sent in data:
                # loop on sentences (comments?)
                comment = set(map(lambda f: self.vocab[f], sent))
                labeled = False

                for b in bad_keywords:
                    if b in comment:
                        comments_labels.append(1)
                        labeled = True
                        break
                if not labeled:
                    comments_labels.append(0)
                comments_labels_dialect.append( np.argmax(predict.predict( ' '.join(list(comment)) )) )

            # add to self
            self.comments_labels.append(comments_labels)
            self.comments_labels_dialect.append(comments_labels_dialect)

            # set the data group
            group = 1 if sum(comments_labels) > len(comments_labels) / 2 else 0
            self.group.append(group)
            
            freq = {x:comments_labels_dialect.count(x) for x in comments_labels_dialect}
            self.group_dialect.append(max(freq, key=freq.get))    

    def __getitem__(self, idx, comments_details=False):
        max_comments = self.max_comments
        comments = self.data[idx][0][:max_comments]
        num_sents = min(len(comments), max_comments)
        num_words = [min(len(sent), 150) for sent in comments][:max_comments]

        if not comments_details:
            return comments, self.labels[idx], num_sents, num_words, self.group[idx], self.group_dialect[idx]
        return comments, self.labels[idx], num_sents, num_words, self.comments_labels[idx], self.comments_labels_dialect[idx]
        

    def __len__(self):
        return len(self.labels)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def num_classes(self):
        return 2


###### Jigsaw DATASET ######
class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, loading_mode='train', comments_aae=False, pre_loaded_data=None, max_comments=None, word_level=False, window_size=None, offset=None):
        MAX_SENT_LENGTH = 150
        self.max_comments = max_comments
        MAX_SENTS = 1000
        self.window_size = window_size
        self.offset = offset

        if window_size != None and offset == None:
            raise ValueError

        if pre_loaded_data != None:
            # load from array!
            self.labels, self.data, self.z_groups = pre_loaded_data[0], pre_loaded_data[1], pre_loaded_data[2]
            self.vocab = pd.read_csv(
                filepath_or_buffer='data/glove/glove.6B.100d.txt',
                header=None,
                sep=" ",
                quoting=csv.QUOTE_NONE,
                usecols=[0]).values[:50000]
            self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
            self.comments_aae = comments_aae

            if comments_aae:
                self.aae_model = predict.load_model()
            else:
                self.aae_model = None

            puncs_table = str.maketrans(dict.fromkeys(string.punctuation))
            return
        
        if comments_aae:
            self.aae_model = predict.load_model()
        else:
            self.aae_model = None

        # save sensitive attributes
        self.gender_atts = ['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']
        self.race_atts = ['black', 'white', 'asian', 'latino', 'other_race_or_ethnicity']
        self.useful_attss = self.gender_atts + self.race_atts
        self.z_groups = []

        dataset = pd.read_csv(datapath)

        # filter the dataset and remove rows containing null values
        # if loading_mode == 'train':
        #    dataset.dropna(inplace=True)
        #    dataset.reset_index(inplace=True)

        self.vocab = pd.read_csv(
            filepath_or_buffer='data/glove/glove.6B.100d.txt',
            header=None,
            sep=" ",
            quoting=csv.QUOTE_NONE,
            usecols=[0]).values[:50000]
        self.vocab = ['<pad>', '<unk>'] + [word[0] for word in self.vocab]
        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        # split into sentences and words
        if not os.path.exists('./data/jigsaw_cache_%s_%s.pkl' %(loading_mode, word_level)):
            self.data = []
            self.labels = []
            labels_count = {0: 0, 1: 0}

            for idx in tqdm(range(len(dataset))):
                # check if the data is valid for our task
                c_flag = False

                for col_name in self.useful_attss:
                    if np.isnan(dataset.loc[idx, col_name]):
                        c_flag = True
                        break
                if c_flag:
                    continue


                if word_level:
                    comments = dataset.loc[idx, 'comment_text']
                else:
                    comments = [dataset.loc[idx, 'comment_text']]

                doc = [
                    [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
                    for sent in comments if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
                doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
                num_sents = min(len(doc), MAX_SENTS)

                if num_sents == 0:
                    continue

                num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

                # if loading_mode == 'train' and dataset.loc[idx, 'target'] < 0.5 and labels_count[0] > 25000:
                #     continue

                if loading_mode == 'train':
                    corrected_label = 0 if dataset.loc[idx, 'target'] < 0.5 else 1
                    self.labels.append(corrected_label)
                    labels_count[corrected_label] += 1
                else:
                    self.labels.append(-1)

                self.data.append((
                    doc,
                    num_sents,
                    num_words
                ))

                if loading_mode == 'test':
                    self.z_groups.append((
                        -1,
                        -1
                    ))
                    continue

                # define groups here -> do not need to create comments labels anymore!
                data = dataset.loc[idx]
                z_gender = data[self.gender_atts].astype('float64').argmax()

                if z_gender > 2:
                    z_gender = 2

                z_race = data[self.race_atts].astype('float64').argmax()
                self.z_groups.append((
                    z_gender,
                    z_race
                ))

            pickle.dump([self.labels, self.data, self.z_groups], open('./data/jigsaw_cache_%s_%s.pkl' %(loading_mode, word_level), 'wb'))
        else:
            print('Loading Data From Cache :)')
            self.labels, self.data, self.z_groups = pickle.load(open('./data/jigsaw_cache_%s_%s.pkl' %(loading_mode, word_level), 'rb'))

    def preprocess_text(self, text):
        for c in ['\'', '"', '@', ',', '#', '%%', '.', '!']:
            text = text.replace(c, '')
        return text.strip().lower()

    def transform(self, comments_list, remove_puncs=True):
        MAX_SENT_LENGTH = 150
        MAX_SENTS = 40 if self.max_comments == None else self.max_comments
        puncs_table = str.maketrans(dict.fromkeys(string.punctuation))

        doc = [
            [self.vocab.index(word) if word in self.vocab else 1 for word in word_tokenize(self.preprocess_text(sent.translate(puncs_table)))]
            for sent in comments_list if len(self.preprocess_text(sent.translate(puncs_table))) > 0]
        doc = [sent[:MAX_SENT_LENGTH] for sent in doc][:MAX_SENTS]
        num_sents = min(len(doc), MAX_SENTS)

        if num_sents == 0:
            return None, 0, None

        num_words = [min(len(sent), MAX_SENT_LENGTH) for sent in doc][:MAX_SENTS]

        return doc, num_sents, num_words, -1

    def get_perspective_comment_labels(self):
        if not os.path.exists('vine_pers_cache.pkl'):
            API_KEY = 'AIzaSyAv6aH0IbcCweqJjM3JoV_M3vqszQdmAa0'

            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

            self.comments_pers_labels = []

            # set labels according to the keywords
            for data, _, _ in tqdm(self.data):
                # loop on sentences
                session_comments_labels = []
                for sent in data:
                    start_time = time.time()
                    # loop on sentences (comments?)
                    comment = ' '.join(list(map(lambda f: self.vocab[f], sent)))
                    
                    # get the comment label
                    analyze_request = {
                        'comment': { 'text': comment },
                        'requestedAttributes': {'TOXICITY': {}},
                        'languages': ['en']
                    }
                    response = client.comments().analyze(body=analyze_request).execute()

                    session_comments_labels.append( 
                        response['attributeScores']['TOXICITY']['summaryScore']['value']
                    )

                    end_time = time.time()
                    time.sleep(1.1 - (start_time - end_time))

                # add to self
                self.comments_pers_labels.append(session_comments_labels)
            pickle.dump(self.comments_pers_labels, open('vine_pers_cache.pkl', 'wb'))
        else:
            print('Pers results from cache!')
            self.comments_pers_labels = pickle.load(open('vine_pers_cache.pkl', 'rb'))

    def create_comment_labels(self, keywords_file='./bad-keywords.txt'):
        # use keyword's file to label each comment
        self.comments_labels = []
        self.comments_labels_dialect = []
        self.group = []
        self.group_dialect = []

        bad_keywords = []
        with open(keywords_file, 'r') as fin:
            for line in fin.readlines():
                bad_keywords.append(line.strip().lower())
            fin.close()

        # set labels according to the keywords
        for data, _, _ in self.data:
            # loop on sentences
            comments_labels, comments_labels_dialect = [], []
            for sent in data:
                # loop on sentences (comments?)
                comment = set(map(lambda f: self.vocab[f], sent))
                labeled = False

                for b in bad_keywords:
                    if b in comment:
                        comments_labels.append(1)
                        labeled = True
                        break
                if not labeled:
                    comments_labels.append(0)
                comments_labels_dialect.append( np.argmax(predict.predict( ' '.join(list(comment)) )) )

            # add to self
            self.comments_labels.append(comments_labels)
            self.comments_labels_dialect.append(comments_labels_dialect)

            # set the data group
            group = 1 if sum(comments_labels) > len(comments_labels) / 2 else 0
            self.group.append(group)
            
            freq = {x:comments_labels_dialect.count(x) for x in comments_labels_dialect}
            self.group_dialect.append(max(freq, key=freq.get))    

    def __getitem__(self, idx, comments_details=False):
        if self.window_size == None or self.offset + self.window_size >= len(self.data[idx][0]):
            max_comments = min(len(self.data[idx][0]), self.max_comments)
            comments = self.data[idx][0][:max_comments]
            num_sents = min(len(comments), max_comments)
            num_words = [min(len(sent), 150) for sent in comments][:max_comments]
        elif self.offset + self.window_size < len(self.data[idx][0]):
            max_comments = min(len(self.data[idx][0]), self.window_size)
            comments = self.data[idx][0][self.offset : self.offset + self.window_size]
            num_sents = min(len(comments), max_comments)
            num_words = [min(len(sent), 150) for sent in comments][:max_comments]
    
        if comments_details:
            return comments, self.labels[idx], num_sents, num_words, [self.z_groups[idx][0]], [self.z_groups[idx][1]]
        return comments, self.labels[idx], num_sents, num_words, self.z_groups[idx][0], self.z_groups[idx][1]

    def __len__(self):
        return len(self.labels)

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def num_classes(self):
        return 2

