import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os
import PyPDF2
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, words, wordnet as wn
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import ssl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- SSL Fix ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Download NLTK Resources ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# --- Helper Functions ---
def is_verb_or_adjective(word):
    synsets = wn.synsets(word)
    if not synsets:
        return False
    return any(s.pos() in ['v', 'a'] for s in synsets)

def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def preprocess(text, custom_stopwords=None):
    english_words = set(words.words())
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and len(token) > 1]
    tokens = [token for token in tokens if token in english_words]
    tokens = [token for token in tokens if not is_verb_or_adjective(token)]
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    if custom_stopwords:
        tokens = [token for token in tokens if token not in custom_stopwords]
    return tokens

# --- Preprocess PDFs ---
pdf_directory = '/Users/apple/Downloads/Construction Safety and EEG'
documents = []
pdf_filenames = []

if not os.path.exists(pdf_directory):
    raise ValueError(f"Directory {pdf_directory} does not exist.")

for pdf_file in os.listdir(pdf_directory):
    if pdf_file.lower().endswith(".pdf"):
        pdf_filenames.append(pdf_file)
        print(f"Loaded PDF: {pdf_file}")

custom_stopwords = ['et', 'al', 'fig', 'data','china', 'usa', 'table', 'vol', 'journal', 'also','sa', 'two', 'one', 'among']

for filename in pdf_filenames:
    pdf_path = os.path.join(pdf_directory, filename)
    text = read_pdf(pdf_path)
    tokens = preprocess(text, custom_stopwords)
    documents.append(tokens)

# --- LDA Setup ---
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

random_state = 35
lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10, random_state=random_state)

coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='u_mass')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score (u_mass): {coherence_score}")

print(f"Number of PDFs: {len(pdf_filenames)}")
print(f"Number of Documents in Corpus: {len(corpus)}")

if len(pdf_filenames) != len(corpus):
    raise ValueError("Mismatch between the number of PDFs and the corpus size.")
assert len(pdf_filenames) == len(corpus), "Mismatch between the number of PDFs and corpus size."

# --- Display Topics ---
for idx, topic in lda_model.print_topics(num_words=8):
    print(f"Topic {idx}: {topic}")

# --- pyLDAvis Output ---
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_vis, "LDAvis_output.html")

# --- Assign Most Probable Topic per Document ---
for i, topic_dist in enumerate(lda_model.get_document_topics(corpus)):
    most_probable_topic = max(topic_dist, key=lambda x: x[1])[0]
    print(f"Document: {pdf_filenames[i]}")
    print(f"Most Probable Topic: {most_probable_topic}")
    print("-" * 30)

# --- Topic Correlation Matrix (Cosine Similarity based on topic-word distributions) ---
topic_term_matrix = lda_model.get_topics()
topic_similarity = cosine_similarity(topic_term_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(topic_similarity, annot=True, cmap="coolwarm",
            xticklabels=[f"T{i}" for i in range(len(topic_similarity))],
            yticklabels=[f"T{i}" for i in range(len(topic_similarity))])
plt.title("Topic Correlation Matrix (Cosine Similarity)")
plt.tight_layout()
plt.savefig("topic_correlation_matrix.png")
plt.show()
