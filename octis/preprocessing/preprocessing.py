import string
from typing import List, Union

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map  # or thread_map
from tqdm import tqdm
from pathlib import Path
from octis.dataset.dataset import Dataset
from collections import Counter

from pandas import read_csv


"""
Maps the language to its corresponding spacy model
"""
spacy_model_mapping = {
    'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm',
    'dutch': 'nl_core_news_sm', 'english': 'en_core_web_sm',
    'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
    'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm',
    'japanese': 'ja_core_news_sm', 'lithuanian': 'lt_core_news_sm',
    'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
    'portuguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm',
    'russian': 'ru_core_news_sm', 'spanish': 'es_core_news_sm'}


class Preprocessing:
    def __init__(
        self, lowercase: bool = True, vocabulary: List[str] = None,
        max_features: int = None, min_df: float = 0.0, max_df: float = 1.0,
        remove_punctuation: bool = True, punctuation: str = string.punctuation,
        remove_numbers: bool = True, lemmatize: bool = True,
        stopword_list: Union[str, List[str]] = None, min_chars: int = 1,
        min_words_docs: int = 0, language: str = 'english', split: bool = True,
        verbose: bool = False, num_processes: int = None,
        save_original_indexes=True, remove_stopwords_spacy: bool = True,
        train_prop: float = 14/20, test_prop: float = 3/20, val_prop: float = None):
        """
        init Preprocessing

        :param lowercase: if true, words in documents are reduced to
            lowercase (default: true)
        :type lowercase: boolean
        :param vocabulary: the vocabulary of the corpus to preprocess
            (default: None)
        :type vocabulary: list
        :param max_features: maximum number of words that the vocabulary must
            contain. The less frequent words will be removed. If it's not None,
            then max_df and min_df are ignored (default: None)
        :type max_features: int
        :param min_df: words below this minumum document frequency will be. It should always be decimal (0.0 not 0)
            removed (default: 0.0)
        :type min_df: float
        :param max_df: words above this maximum document frequency will beIt should always be decimal (1.0 not 1)
            removed (default: 1.0)
        :type max_df: float
        :param remove_punctuation: if true, punctuation will be removed
            (default: true)
        :type remove_punctuation: bool
        :param punctuation: string containing all the punctuation chars that
            need to be removed (default:
        string.punctuation)
        :type punctuation: str
        :param remove_numbers: if true, numbers will be removed
        :type remove_numbers: bool
        :param remove_stopwords_spacy: bool , if true use spacy to remove
            stopwords (default: true)
        :param lemmatize: if true, words will be lemmatized using a spacy model
            according to the language that has been set (default: true)
        :type lemmatize: bool
        :param stopword_list: if a list of strings is passed, the strings will
            be removed from the texts. Otherwise, if a str is passed, it
            represents the language of the stopwords that need to be removed.
            The stopwords are spacy's stopwords (default: None)
        :type stopword_list: str or list of str
        :param min_chars: mininum number of characters that a token should have
            (default: 1)
        :type min_chars: int
        :param min_words_docs: minimun number of words that a document should
            contain (default: 0)
        :type min_words_docs: int
        :param language: language of the documents. It needs to be set for the
            lemmatizer (default: english)
        :type language: str
        :param split: if true, the corpus will be split in train (85%),
            testing (7.5%) and validation (7.5%) set (default: true)
        :type split: bool
        :param verbose: if true, some steps of the preprocessing will be
            printed on screen (default: false)
        :type verbose: bool
        :param num_processes: number of processes to run the preprocessing
        :type num_processes: int
        :param save_original_indexes: if true, it keeps track of the original
            indexes of the documents
        :param train_prop: ratio of data in training
        :type train_prop: float
        :param test_prop: ratio of data in testing
        :type test_prop: float
        :param val_prop: ratio of data in validation
        :type val_prop: float
        """
        self.vocabulary = vocabulary
        self.lowercase = lowercase
        self.max_features = max_features
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.remove_punctuation = remove_punctuation
        self.punctuation = punctuation
        self.lemmatize = lemmatize
        self.language = language
        self.num_processes = num_processes
        self.remove_numbers = remove_numbers
        self.save_original_indexes = save_original_indexes

        if (train_prop is not None) & (test_prop is not None):
            val_prop = 1 - train_prop - test_prop
        elif (train_prop is not None) & (val_prop is not None):
            test_prop = 1 - train_prop - val_prop
        elif (test_prop is not None) & (val_prop is not None):
            train_prop = 1 - test_prop- val_prop

        self.first_ratio = test_prop
        self.second_ratio = 1/(1 + train_prop/val_prop)

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        self.split = split
        self.verbose = verbose

        self.remove_stopwords_spacy = remove_stopwords_spacy

        stopwords = []
        # if stopwords is None then stopwords are not removed
        if stopword_list is None:
            self.remove_stopwords_spacy = False
        else:
            # if custom list is specified, then we do not use spacy stopwords
            if type(stopword_list) == list:
                stopwords = set(stopword_list)
                self.remove_stopwords_spacy = False
            elif self.remove_stopwords_spacy:
                assert stopword_list == language
            else:
                # if remove_stopwords_spacy is false, then use MALLET English stopwords
                if 'english' in stopword_list:
                    stop_word_path = Path(__file__).parent.joinpath('stopwords', 'english.txt')
                    with open(stop_word_path) as fr:
                        stopwords = [line.strip() for line in fr.readlines()]
                        assert stopword_list == language

        self.stopwords = stopwords
        self.min_chars = min_chars
        self.min_doc_words = min_words_docs
        self.preprocessing_steps = []

    def preprocess_dataset(self, documents_path, labels_path=None, multilabel=False):
        #all the documents are inside the documents_path file, that is a .txt file

        """
        preprocess the input dataset

        :param documents_path: path to the documents file. Each row of the file represents a document
        :type documents_path: str
        :param labels_path: path to the documents file. Each row of the file represents a label. Its index corresponds
        to the index of the documents file (default: None)
        :type labels_path: str
        :param multilabel: if true, a document is supposed to have more than one label (labels are split by whitespace)
        :type multilabel: bool

        :return octis.dataset.dataset.Dataset


        details:
        If self.split is true, train-validation-test split is performed, and
        the rows of the documents are reordered accordingly to this criteria:
        first train, then validation, then test
        The metadata attributes
        "last-training-doc", "last-validation-doc", "total_documents"
        reflects the boundaries of the partition, so that is possible to rebuild the splits from 
        the Dataset class.
        The document_indexes are the original indexes of the dataset.
        If self.save_original_indexes is true, then they are passed to Dataset()
        """
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]
        if self.num_processes is not None:
            # with Pool(self.num_processes) as p:
            #    docs = p.map(self.simple_preprocessing_steps, docs)
            chunksize = max(1, len(docs) // (self.num_processes * 20))
            docs_list = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=chunksize)
        else:
            #print('simple prepro started')
            docs = list(map(self.simple_preprocessing_steps, tqdm(docs)))
            #print('simple prepro ended')
        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')

        vocabulary = self.filter_words(docs)
        print("created vocabulary with " + str(len(vocabulary)) + ' words')
        final_docs, final_labels, document_indexes = [], [], []
        if labels_path is not None:
            if multilabel:
                labels = [
                    line.strip().split()
                    for line in open(labels_path, 'r').readlines()]
            else:
                labels = [
                    line.strip()
                    for line in open(labels_path, 'r').readlines()]

            vocab = set(vocabulary)
            for i, doc, label in zip(range(len(docs)), docs, labels):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
                    document_indexes.append(i)

            labels_to_remove = set([k for k, v in dict(
                Counter(final_labels)).items() if v <= 3])
            if len(labels_to_remove) > 0:
                docs = final_docs
                labels = final_labels
                document_indexes, final_labels, final_docs = [], [], []
                for i, doc, label in zip(range(len(docs)), docs, labels):
                    if label not in labels_to_remove:
                        final_docs.append(doc)
                        final_labels.append(label)
                        document_indexes.append(i)
        else:
            vocab = set(vocabulary)
            for i, doc in enumerate(docs):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    document_indexes.append(i)

        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps
                    # ,"labels": list(set(final_labels)), "total_labels": len(set(final_labels))
                    }
        if self.split:
            if len(final_labels) > 0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size = self.first_ratio,#0.15, 
                    random_state=1, shuffle=True)#stratify=final_labels)

                train, validation = train_test_split(train, test_size = self.second_ratio,  #3 / 17, 
                                                     random_state=1, shuffle=True)# stratify=y_train)

                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels, document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                   labels=partitioned_labels)
            else:
                train, test = train_test_split(range(len(final_docs)), test_size = self.first_ratio, #0.15, 
                                               random_state=1)
                train, validation = train_test_split(train, test_size = self.second_ratio, #3 / 17, 
                                                     random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                   document_indexes=document_indexes)
        else:
            if self.save_original_indexes:
                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                               document_indexes=document_indexes)
            else:

                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels)

    def filter_words(self, docs):
        if self.vocabulary is not None:
            self.preprocessing_steps.append('filter words by vocabulary')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(df_max_freq=self.max_df, df_min_freq=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            self.preprocessing_steps.append('filter vocabulary to ' + str(self.max_features) + ' terms')
            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            # we ignore df_max_freq e df_min_freq because self.max_features is not None
            vectorizer = TfidfVectorizer(lowercase=self.lowercase, max_features=self.max_features,
                                         stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b")

        else:

            #string.punctuation

            self.preprocessing_steps.append('filter words with document frequency lower than ' + str(self.min_df) +
                                            ' and higher than ' + str(self.max_df))
            self.preprocessing_steps.append('filter words with less than ' + str(self.min_chars) + " character")
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        # print('vectorizer called')
        # print(len(docs))
        # print(type(docs))
        # print(type(docs[0]))
        # print(docs[0])
        # print(docs[1])

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names_out()
        return vocabulary

    '''
    def _foo(self, docs, vocabulary, labels_path):
        final_docs, final_labels = [], []
        if labels_path is not None:
            labels = [line.strip() for line in open(labels_path, 'r').readlines()]
            for doc, label in zip(docs, labels):
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
            return final_docs, final_labels
        else:
            for doc in docs:
                new_doc = [w for w in doc.split() if w in set(vocabulary)]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
            return final_docs, []
    '''

    def simple_preprocessing_steps(self, doc):
        new_d = doc
        new_d = new_d.replace('\n', '')
        new_d = new_d.replace('\t', '')
        if self.lowercase:
            new_d = new_d.lower()
        if self.lemmatize:
            if self.remove_stopwords_spacy:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if not token.is_stop])
            elif self.stopwords:
                new_d = ' '.join(
                    [token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
            else:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])

        if self.remove_punctuation:
            new_d = new_d.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))
        if self.remove_numbers:
            new_d = new_d.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
        new_d = " ".join(new_d.split())
        return new_d




    def preprocess_csv_dataset(self, path, sep=';', labels_sep = ' ',
                              docs_colname=None, 
                              labels_colname=None, 
                              multilabel=False):
        """
        preprocess the input dataset

        :param path: column of the documents file. Each cell of the column represents a document
        :type documents_path: str
        :param labels_colname: column of the documents. Each cell of the column represents a label. Its index corresponds
        to the index of the documents file (default: None). 
        :type labels_path: str
        :param multilabel: if true, a document is supposed to have more than one label (labels are split by whitespace by default)
        :type multilabel: bool

        :return octis.dataset.dataset.Dataset

        details:
        If self.split is true, train-validation-test split is performed, and
        the rows of the documents are reordered accordingly to this criteria:
        first train, then validation, then test
        The metadata attributes
        "last-training-doc", "last-validation-doc", "total_documents"
        reflects the boundaries of the partition, so that is possible to rebuild the splits from 
        the Dataset class.
        The document_indexes are the original indexes of the dataset.
        If self.save_original_indexes is true, then they are passed to Dataset()
        """

        tab_data = read_csv(path, sep=sep)  #from pandas import read_csv
        docs = list(tab_data[docs_colname])

        if self.num_processes is not None:
            # with Pool(self.num_processes) as p:
            #    docs = p.map(self.simple_preprocessing_steps, docs)
            chunksize = max(1, len(docs) // (self.num_processes * 20))
            docs_list = process_map(self.simple_preprocessing_steps, docs, max_workers=self.num_processes, chunksize=chunksize)
        else:
            print('simple prepro started')
            docs = list(map(self.simple_preprocessing_steps, tqdm(docs)))
            print('simple prepro ended')
        if self.lowercase:
            self.preprocessing_steps.append("lowercase")
        if self.remove_punctuation:
            self.preprocessing_steps.append('remove_punctuation')
        if self.lemmatize:
            self.preprocessing_steps.append('lemmatize')


        vocabulary = self.filter_words(docs)
        print("created vocab")
        print(len(vocabulary))
        final_docs, final_labels, document_indexes = [], [], []



        if labels_colname is not None:

            print('there are labels with texts')
            if multilabel:
                labels = list(tab_data[labels_colname])
                labels = [line.strip().split() for line in labels]
            else:
                labels = list(tab_data[labels_colname])

            vocab = set(vocabulary)
            filtered_out = 0
            for i, doc, label in zip(range(len(docs)), docs, labels):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    final_labels.append(label)
                    document_indexes.append(i)
                else:
                    filtered_out += 1
            print("Filtered out docs:", filtered_out)
            print('num of docs: ' + str(len(final_docs)))

            labels_to_remove = set([k for k, v in dict(
                Counter(final_labels)).items() if v <= 3])
            
            print('num of labels to remove: ' + str(len(labels_to_remove)))

            if len(labels_to_remove) > 0:
                print('some labels will be removed')
                docs = final_docs
                labels = final_labels
                document_indexes, final_labels, final_docs = [], [], []
                for i, doc, label in zip(range(len(docs)), docs, labels):
                    if label not in labels_to_remove:
                        final_docs.append(doc)
                        final_labels.append(label)
                        document_indexes.append(i)
            
                print('num of docs: ' + str(len(final_docs)))
        else:
            vocab = set(vocabulary)
            for i, doc in enumerate(docs):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_doc_words:
                    final_docs.append(new_doc)
                    document_indexes.append(i)

        self.preprocessing_steps.append('filter documents with less than ' + str(self.min_doc_words) + " words")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary),
                    "preprocessing-info": self.preprocessing_steps
                    # ,"labels": list(set(final_labels)), "total_labels": len(set(final_labels))
                    }
        if self.split:
            if len(final_labels) > 0:
                train, test, y_train, y_test = train_test_split(
                    range(len(final_docs)), final_labels, test_size = self.first_ratio,#0.15, 
                    random_state=1, shuffle=True)#stratify=final_labels)

                train, validation = train_test_split(train, test_size = self.second_ratio, #3 / 17, 
                                                     random_state=1, shuffle=True)# stratify=y_train)

                partitioned_labels = [final_labels[doc] for doc in train + validation + test]
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                    labels=partitioned_labels, document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata,
                                    labels=partitioned_labels)
            else:
                train, test = train_test_split(range(len(final_docs)), test_size=0.15, random_state=1)
                train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

                metadata["last-training-doc"] = len(train)
                metadata["last-validation-doc"] = len(validation) + len(train)
                partitioned_corpus = [final_docs[doc] for doc in train + validation + test]
                document_indexes = [document_indexes[doc] for doc in train + validation + test]
                if self.save_original_indexes:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                    document_indexes=document_indexes)
                else:
                    return Dataset(partitioned_corpus, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                    document_indexes=document_indexes)
        else:
            if self.save_original_indexes:
                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                document_indexes=document_indexes)
            else:

                return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels)
