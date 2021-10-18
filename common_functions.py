import json
import numpy as np
from nltk import word_tokenize
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def norm(vec):
    return vec / np.linalg.norm(vec)


def preprocess(text):
    tokens = [token for token in word_tokenize(text.lower())
              if token.isalpha()]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def collect_answers(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = list(f)[:10000]  # 10 000
    proc_answers, raw_answers = [], []

    for question in data:
        item = json.loads(question)
        if item['answers']:
            ids = [int(answer['author_rating']['value']) for answer in item['answers']
                   if answer['author_rating']['value']]
            id_max = max(zip(ids, range(len(ids))))[1]
            answer = item['answers'][id_max]['text']
            proc_answer = preprocess(answer)
            proc_question = preprocess(item['question'])
            # проверка, что и вопрос, и ответ не пустые после лемматизации
            if (proc_answer != '') & (proc_question != ''):
                proc_answers.append(proc_answer)
                raw_answers.append(answer)
    # сохранить в файл
    proc_data = {'proc_answers': proc_answers,
                 'raw_answers': raw_answers}
    with open('data/processed_corpus.json', 'w', encoding='UTF-8') as new_file:
        json.dump(proc_data, new_file, ensure_ascii=False)


def count_sim(matrix, query_vector):
    scores = np.dot(matrix, query_vector.T)
    return scores


def count_sim_sparse(matrix, query_vector):
    scores = matrix @ query_vector.T
    return scores.toarray()
