try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _USE_SKLEARN = True
except Exception:
    _USE_SKLEARN = False

import re
import math
import sys

faq = {
    "What is Python?": "Python is a programming language.",
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine Learning enables computers to learn from data."
}

questions = list(faq.keys())

# export safe placeholders so importing the module won't fail
vectorizer = None
vectors = None
doc_vectors = None
_idf = {}

def _tokenize(text):
    return re.findall(r"\w+", text.lower())

def _build_tfidf(docs):
    N = len(docs)
    df = {}
    docs_tokens = []
    for d in docs:
        toks = set(_tokenize(d))
        docs_tokens.append(toks)
        for t in toks:
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log((N + 1) / (df.get(t, 0) + 1)) + 1 for t in df}

    vectors = []
    for d in docs:
        tf = {}
        toks = _tokenize(d)
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (tf.get(t, 0) * idf.get(t, 0)) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for t in vec:
                vec[t] /= norm
        vectors.append(vec)

    return vectors, idf

def _cosine_sim(query_vec, doc_vec):
    # both are dicts term->value
    return sum(query_vec.get(t, 0) * v for t, v in doc_vec.items())

if _USE_SKLEARN:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(questions)

    def _most_similar(query):
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, vectors)[0]
        return sims.argmax()
else:
    doc_vectors, _idf = _build_tfidf(questions)
    stopwords = {"what", "is", "the", "a", "an", "of", "in", "to"}

    def _most_similar(query):
        toks = [t for t in _tokenize(query) if t not in stopwords]
        # simple overlap matching first
        best = 0
        best_score = -1
        qset = set(toks)
        for i, q in enumerate(questions):
            q_tokens = set([t for t in _tokenize(q) if t not in stopwords])
            score = len(qset & q_tokens)
            if score > best_score:
                best_score = score
                best = i

        if best_score > 0:
            return best

        # fallback to TF-IDF similarity
        tf = {}
        for t in _tokenize(query):
            tf[t] = tf.get(t, 0) + 1
        qvec = {t: (tf[t] * _idf.get(t, math.log((len(questions) + 1) / 1 + 1))) for t in tf}
        norm = math.sqrt(sum(v * v for v in qvec.values()))
        if norm > 0:
            for t in qvec:
                qvec[t] /= norm

        sims = [ _cosine_sim(qvec, dv) for dv in doc_vectors ]
        best = 0
        best_s = -1
        for i, s in enumerate(sims):
            if s > best_s:
                best_s = s
                best = i
        return best


def main():
    debug = '--debug' in sys.argv
    while True:
        try:
            query = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() == "exit":
            break

        index = _most_similar(query)
        if debug:
            # show debug info
            if _USE_SKLEARN:
                from sklearn.metrics.pairwise import cosine_similarity
                qv = vectorizer.transform([query])
                sims = list(cosine_similarity(qv, vectors)[0])
                print('DEBUG sims:', sims)
            else:
                toks = _tokenize(query)
                qset = set([t for t in toks if t not in {"what","is","the","a","an","of","in","to"}])
                overlaps = []
                for i, qq in enumerate(questions):
                    q_tokens = set([t for t in _tokenize(qq) if t not in {"what","is","the","a","an","of","in","to"}])
                    overlaps.append(len(qset & q_tokens))
                print('DEBUG overlaps:', overlaps)
                tf = {}
                for t in toks:
                    tf[t] = tf.get(t, 0) + 1
                qvec = {t: (tf[t] * _idf.get(t, 1.0)) for t in tf}
                norm = math.sqrt(sum(v * v for v in qvec.values()))
                if norm > 0:
                    for t in qvec:
                        qvec[t] /= norm
                sims = [ _cosine_sim(qvec, dv) for dv in doc_vectors ]
                print('DEBUG tfidf sims:', sims)

        print("Bot:", faq[questions[index]])

if __name__ == "__main__":
    main()