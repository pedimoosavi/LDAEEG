import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import os
import PyPDF2
import nltk
from nltk.corpus import stopwords, words, wordnet as wn
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context



# Ensure you have the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

# Function to check if a word is a verb or adjective
def is_verb_or_adjective(word):
    synsets = wn.synsets(word)
    if not synsets:
        return False
    return any(s.pos() in ['v', 'a'] for s in synsets)

# Function to read PDF and extract text
def read_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
    return text

# Function to preprocess the text and exclude specified words
def preprocess(text, custom_stopwords=None):
    english_words = set(words.words())  # Load the set of English words
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and len(token) > 1] # Exclude single letter words
    tokens = [token for token in tokens if token in english_words]  # Keep only English words
    tokens = [token for token in tokens if not is_verb_or_adjective(token)]  # Exclude verbs and adjectives
    tokens = [token for token in tokens if token not in stopwords.words('english')]

    if custom_stopwords:
        tokens = [token for token in tokens if token not in custom_stopwords]
        
    return tokens

# Load and preprocess PDF files from a directory
pdf_directory = '/Users/apple/Downloads/Construction Safety and EEG'
documents = []
pdf_filenames = []

# Ensure the directory exists and is accessible
if not os.path.exists(pdf_directory):
    raise ValueError(f"Directory {pdf_directory} does not exist.")

# Populate pdf_filenames
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.lower().endswith(".pdf"):  # Consider all cases of .pdf extension
        pdf_filenames.append(pdf_file)  # Append only the filename, not the full path
        print(f"Loaded PDF: {pdf_file}")

# Define a list of custom stopwords to exclude from the analysis
custom_stopwords = ['et', 'al', 'fig', 'data','china', 'usa', 'table', 'vol', 'journal', 'also','sa', 'two', 'one', 'among']

for filename in pdf_filenames:
    pdf_path = os.path.join(pdf_directory, filename)
    text = read_pdf(pdf_path)
    tokens = preprocess(text, custom_stopwords=custom_stopwords)
    documents.append(tokens)

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Set the random seed for reproducibility
#make a spreadsheat based on different random states, and coherence scores

random_state = 35

# Train the LDA model with a fixed random seed
lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10, random_state=random_state)

coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='u_mass')
coherence_score = coherence_model.get_coherence()
print(f"Coherence Score (u_mass): {coherence_score}")

# Print lengths to debug
print(f"Number of PDFs: {len(pdf_filenames)}")
print(f"Number of Documents in Corpus: {len(corpus)}")

# Check if the lengths of pdf_filenames and corpus are consistent
if len(pdf_filenames) != len(corpus):
    raise ValueError("Mismatch between the number of PDFs and the corpus size.")
assert len(pdf_filenames) == len(corpus), "Mismatch between the number of PDFs and corpus size."

for idx, topic in lda_model.print_topics(num_words=8):
    print(f"Topic {idx}: {topic}")
    
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(lda_vis, "LDAvis_output.html")

# Assign topics to each pdf and print the most probable topic for each
for i, topic_dist in enumerate(lda_model.get_document_topics(corpus)):
    most_probable_topic = max(topic_dist, key=lambda x: x[1])[0]
    print(f"Document: {pdf_filenames[i]}")
    print(f"Most Probable Topic: {most_probable_topic}")
    print("-" * 30)
