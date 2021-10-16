'''
логика файлов:
5 файлов под каждый из пяти способов поиска.
каждый файл кончается собиранием всего в сёрч/старт функцию
отд. файл под общие функции: нормализация, препроцессинг,
собирание данных из файла, поиск близости, сортировка

в main файле весь бэк + вызов нужной функции в зависимости от
выбранного способа

в файлы способов поиска надо добавить сохранение текстов,
матриц и векторайзеров а затем их считывание
'''


import json
import shutup
shutup.please()
import time
import numpy as np
import streamlit as st

from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

from common_functions import collect_answers, preprocess
from count_functions import index_corpus_count, search_count
from tfidf_functions import index_corpus_tfidf, search_tfidf
from bm25_functions import index_corpus_bm25, search_bm25
from fasttext_functions import index_corpus_fasttext, search_fasttext
from bert_functions import index_corpus_bert, search_bert


def prestart(filename):
    '''загрузить модели, посчитать и создать все нужные файлы'''
    ft_model_path = '../hw4/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    fasttext_model = KeyedVectors.load(ft_model_path)
    bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    bert_model = {'bert_tokenizer': bert_tokenizer,
                  'bert_model': bert_model}
    # collect_answers(filename)
    with open('data/processed_corpus.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    # proc_answers = data['proc_answers']
    raw_answers = data['raw_answers']
    # index_corpus_count(proc_answers)
    # index_corpus_tfidf(proc_answers)
    # index_corpus_bm25(proc_answers)
    # index_corpus_fasttext(proc_answers, fasttext_model)
    # index_corpus_bert(raw_answers, **bert_model)

    return raw_answers, fasttext_model, bert_model


def start(query, vectorizer, raw_answers, fasttext_model, bert_model):
    arr_answers = np.array(raw_answers)
    if vectorizer == 'count':
        proc_query = preprocess(query)
        order = search_count(proc_query)
    elif vectorizer == 'tfidf':
        proc_query = preprocess(query)
        order = search_tfidf(proc_query)
    elif vectorizer == 'bm25':
        proc_query = preprocess(query)
        order = search_bm25(proc_query)
    elif vectorizer == 'fasttext':
        proc_query = preprocess(query)
        order = search_fasttext(proc_query, fasttext_model)
    elif vectorizer == 'bert':
        order = search_bert(query, **bert_model)
    else:
        raise TypeError('Unknown vectorizer')
    sorted_answers = arr_answers[order.ravel()]
    return sorted_answers

raw_answers, fasttext_model, bert_model = prestart('data/questions_about_love.jsonl')
#print(start('муж', 'bert', raw_answers, fasttext_model, bert_model))


st.title('поисковик ответов про любовь')
st.header('Что вы хотите узнать?')
st.info('Здесь можно задать самые сокровенные вопросы лучшим умам '
        'человечества, чью мудрость навечно сохранил сайт ответы маил ру. '
        'Попробуйте') # , вам не понравится')


def search_function(raw_answers, fasttext_model, bert_model):
    place_zero = st.empty()
    place_one = st.empty()
    place_two = st.empty()
    place_three = st.empty()
    place_four = st.empty()

    query = ' '
    query = place_zero.text_input('Ваш тревожащий вопрос:', query)
    options = ['count', 'tfidf', 'bm25', 'fasttext', 'bert']
    display = ['Count (считаем слова)',
               'Tf-idf (считаем важные слова)',
               'BM-25 (считаем важные слова по-сложному)',
               'FastText (крутая векторная модель)',
               'BERT (очень крутая векторная модель)']
    vectorizer = place_one.selectbox('Надо выбрать метод поиска:',
                                     options, format_func=lambda x: display[options.index(x)])
    button = place_two.button('Хочу ответ!')

    if button:
        start_time = time.time()
        answers = start(query, vectorizer, raw_answers, fasttext_model, bert_model)
        end_time = time.time()
        time = end_time - start_time
        place_one.caption('Ваш ответ:')
        place_two.write('  \n'.join(answers))
        place_three.caption(f'Поиск сработал за {time} сек')
        new_button = place_four.button('Спросить ещё!')

        if new_button:
            search_function(raw_answers, fasttext_model, bert_model)


search_function(raw_answers, fasttext_model, bert_model)
