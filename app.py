from flask import Flask,render_template,url_for,request
import re
import string
import unicodedata
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
import pickle


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	input = input
	df= pd.read_csv("Questions_cleaned.csv")
	
	##Traitement de l'input pour qu'il soit
	## au bon format
	## Fonctions
	def remove_punctuations(text):
		for punctuation in string.punctuation:
			text = text.replace(punctuation, '')
		return text


	def remove_accented_chars(text):
		new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
		return new_text

	def remove_numbers(text):
		# define the pattern to keep
		pattern = r'[^a-zA-z.,!?/:;\"\'\s# +]' 
		return re.sub(pattern, '', text)


	def remove_extra_whitespace_tabs(text):
		#pattern = r'^\s+$|\s+$'
		pattern = r'^\s*|\s\s*'
		return re.sub(pattern, ' ', text).strip()

	nltk.download('wordnet')

	w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
	lemmatizer = nltk.stem.WordNetLemmatizer()

	def lemmatize_text(text):
		return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
	
	## Application fonctions
	input = remove_punctuations(input)
	input = remove_accented_chars(input)
	input = remove_numbers(input)
	input = remove_extra_whitespace_tabs(input)
	input = contractions.fix(input)
	input = input.lower()
	input = lemmatize_text(input)
	for elt_ in input:
		if elt_ in all_stopwords:
			input.remove(elt_)

	### Bag of Words du jeu d'entraînement
	

	questions['body_title']
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df = 2, stop_words="english")
	tf = tf_vectorizer.fit_transform(questions['body_title']) 

	## Bag of words de l'input
	tf_input = tf_vectorizer.transform(input)

	### Bianrisation des labels
	y = questions["Tags"]

	# Multilabel binarizer for targets
	multilabel_binarizer = MultiLabelBinarizer()
	multilabel_binarizer.fit(y)
	y_binarized = multilabel_binarizer.transform(y)

	## Jeu d'

	nltk.download('stopwords')
	## Faire un commentaire sur le no que j'ai décidé de garder dans le rapport écrit parce qu'il ne s'agit pas de sentiment 
	## analysis
	all_stopwords = stopwords.words('english')

	## Importation du modèle
	rfc_model = pickle.load(open("/content/drive/MyDrive/P05_pestourie_mathilde/deploiement_rfc_final.pkl", 'rb'))


	##prédictions
	y_pred = rfc_model.predict(tf_input)
	y_pred_inversed = multilabel_binarizer\
    .inverse_transform(y_pred)
	def Remove(tuples):
		tuples = [t for t in tuples if t]
		return tuples
	result = Remove(y_pred_inversed)
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)