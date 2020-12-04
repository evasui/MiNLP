# Copyright 2020 The MiNLP Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import regex
import os
import tempfile
import tensorflow as tf
import math
from minlptokenizer.config import configs
from minlptokenizer.lexicon import Lexicon
from minlptokenizer.vocab import Vocab
from minlptokenizer.tag import Tag
from minlptokenizer.exception import MaxLengthException, ZeroLengthException, UnSupportedException, MaxBatchException
from functools import partial
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pwd = os.path.dirname(__file__)


def batch_generator(list_texts, size=configs['tokenizer_limit']['max_batch_size']):
    """
    list generator 用于迭代生成batch
    :param list_texts:待切分的语料列表
    :param size: 每个batch的大小
    :return: 迭代器
    """
    if isinstance(list_texts, list):
        batch_num = math.ceil(len(list_texts) / size)
        for i in range(batch_num):
            yield list_texts[i * size:(i + 1) * size]


class MiNLPTokenizer:
    tokenizer_singleton = None
    temp_folder = tempfile.gettempdir()

    def __init__(self, file_or_list=None, granularity='fine'):
        """
        分词器初始化
        :param file_or_list: 用户自定义词典文件或列表
        :param granularity: 分词粒度参数，fine表示细粒度分词，coarse表示粗粒度分词
        """
        self.__char_dict_path = os.path.join(pwd, configs['vocab_path'])
        self.__pb_model_path = os.path.join(pwd, configs['tokenizer_granularity'][granularity]['model'])
        self.__vocab = Vocab(self.__char_dict_path)
        self.__lexicon = Lexicon(file_or_list)
        with tf.io.gfile.GFile(self.__pb_model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        g = tf.Graph()
        with g.as_default():
            tf.import_graph_def(graph_def, name='')
        tf_config = tf.compat.v1.ConfigProto()
        tf_config.gpu_options.allow_growth = True  # 使用过程中动态申请显存，按需分配
        self.__sess = tf.compat.v1.Session(graph=g, config=tf_config)
        self.__char_ids_input = self.__sess.graph.get_tensor_by_name('char_ids_batch:0')
        self.__factor_input = self.__sess.graph.get_tensor_by_name('factor_batch:0')
        self.__tag_ids = self.__sess.graph.get_tensor_by_name('tag_ids:0')
        for lexicon_file in configs['lexicon_files']:
            self.__lexicon.add_words(os.path.join(pwd, lexicon_file))

    @staticmethod
    def __format_string(ustring):
        """
        全角转半角，多个连续控制符、空格替换成单个空格
        """
        if not ustring.strip():
            raise ZeroLengthException()

        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）转化
                inside_code -= 65248

            rstring += chr(inside_code)
        if len(rstring) > configs['tokenizer_limit']['max_string_length']:
            raise MaxLengthException(len(rstring))
        return regex.sub(r'[\p{Z}\s]+', ' ', rstring.strip())

    @staticmethod
    def cut_batch_in_one_process(file_or_list, granularity, text_batch):
        '''
        多进程分词辅助方法，用于在每个子进程内进行分词
        :param file_or_list: 分词对象
        :param granularity:
        :param text_batch:
        :return:
        '''
        if MiNLPTokenizer.tokenizer_singleton is None:
            MiNLPTokenizer.tokenizer_singleton = MiNLPTokenizer(file_or_list, granularity)
        return MiNLPTokenizer.tokenizer_singleton.cut(text_batch)

    @classmethod
    def cut_batch_multiprocess(cls, text_batch, file_or_list=None, granularity='fine', n_jobs=2):
        """
        使用多进程进行批量分词操作
         :param text_batch: 待分词list
         :param file_or_list:干预词典或文件
         :param granularity:分词粒度
         :param n_jobs:2
        """
        partitions = batch_generator(text_batch, size=configs['tokenizer_limit']['max_batch_size'])
        process_pool = Pool(n_jobs)
        process_cut = partial(cls.cut_batch_in_one_process, file_or_list, granularity)
        try:
            res = process_pool.map(process_cut, partitions)
        except Exception:
            raise Exception
        return res

    @staticmethod
    def cut_from_file(file_path, save_path, file_or_list=None, granularity='fine',
                      batch_size=configs['tokenizer_limit']['max_batch_size'], n_jobs=2):
        """
        文件分词功能，支持从文件中读写
        :param file_path:待分词文件路径
        :param save_path:分词结果保存路径
        :param file_or_list:分词干预文件或列表
        :param granularity:分词粒度
        :param batch_size:分词batch大小默认为512
        :param n_jobs:分词进程数量，默认为2
        :return:
        """
        file = open(file_path, 'r', encoding='utf-8')
        file_save = open(save_path, 'w', encoding='utf-8')
        process_pool = Pool(n_jobs)
        batch_jobs = list()
        batch = list()
        do_cut = partial(MiNLPTokenizer.cut_batch_in_one_process, file_or_list, granularity)
        for index, sentence in enumerate(file):
            if len(batch_jobs) == n_jobs:
                res = process_pool.map(do_cut, batch_jobs)
                for i in res:
                    for j in i:
                        file_save.writelines(' '.join(j) + '\n')
                batch_jobs = list()
            if index > 0 and index % batch_size == 0:
                batch_jobs.append(batch)
                batch = list()
            batch.append(sentence)
        res = process_pool.map(do_cut, batch_jobs)
        for i in res:
            for j in i:
                file_save.writelines(' '.join(j) + '\n')
        file_save.close()
        file.close()

    def cut(self, text_batch):
        '''
        分词函数，支持对一句话或一个列表进行分词
        :param text_batch:待分词列表/句子
        :return: 分词结果
        '''
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        if isinstance(text_batch, list):
            if len(text_batch) > configs['tokenizer_limit']['max_batch_size']:
                raise MaxBatchException(len(text_batch))
            texts = list(map(self.__format_string, text_batch))
            factor = self.__lexicon.product_factor(texts)
            input_char_id = self.__vocab.get_char_ids(texts)
            feed_dict = {
                self.__char_ids_input: input_char_id,
                self.__factor_input: factor
            }
            y_pred_results = self.__sess.run(self.__tag_ids, feed_dict=feed_dict)
            if len(text_batch) == 1:
                return list(map(lambda x, y: self.__tag2words(x, y), texts, y_pred_results))[0]
            else:
                return list(map(lambda x, y: self.__tag2words(x, y), texts, y_pred_results))
        else:
            return UnSupportedException()

    @staticmethod
    def __tag2words(text, y_pred_result):
        words = []
        word = ''
        for idx, ch in enumerate(text):
            word += ch
            tag = y_pred_result[idx]
            if tag == Tag.S.value or tag == Tag.E.value or tag == Tag.X.value:
                words.append(word)
                word = ''
        if word:
            words.append(word)
        return regex.split(r'\s+', ' '.join(words))

    @classmethod
    def set_memmap_folder(cls, path):
        """
        设置memap的路径（cut_batch_multiprocess 时会使用内存映射来进行通信）
        由于Joblib本身编码原因。路径不要含有中文字符
        :param path: memap路径
        :return:
        """
        cls.temp_folder = path

    def set_interfere_factor(self, interfere_factor):
        """
        设置用户词典干预强度，值越大，分词结果越符合词典
        :param interfere_factor: 干预强度，默认值：2
       """
        self.__lexicon.set_interfere_factor(interfere_factor)

    def reset_interfere_factor(self):
        """
        重置用户词典干预强度为默认值：2
        """
        self.__lexicon.reset_interfere_factor()

    def destroy(self):
        self.__sess.close()
