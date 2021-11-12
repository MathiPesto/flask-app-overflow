from flask import Flask,render_template,url_for,request
import pandas as pd
import re
import string
import unicodedata
import requests
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
	input = str(request.form.get('input'))
	questions= pd.read_csv("question_import.csv")
	
	##Traitement de l'input pour qu'il soit

	## Traitement des tags

	questions["Tags"] = questions.Tags.apply(lambda x: x.replace(" '","").replace("'", "").strip())
	questions["Tags"] = questions.Tags.apply(lambda x: x[1:-1].split(","))
	
	special_chars = {}
	for i, _tags in enumerate(questions['Tags']):
		for _i, _tag in enumerate(_tags):
			if '+' in _tag:
				_tags[_i] = re.sub(r'\+', '_plus', _tag)
			elif '#' in _tag: 
				_tags[_i] = re.sub(r'#', '_sharp', _tag)
	
	for i , _tags in enumerate(questions['Tags']):
  
		for _i, _tag in enumerate(_tags):
			if _tag == "c#":
				_tags[_i] = "c_sharp"
			elif _tag == ".net" :
				_tags[_i] = 'dot_net'
	
	tags = questions['Tags'].apply(lambda x: " ".join(x) )

	cv = CountVectorizer(ngram_range=(1,1),analyzer='word', lowercase=False,min_df=1) 


	Tags_train = cv.fit_transform(tags)
	Tags_train=pd.DataFrame(Tags_train.toarray(), columns=cv.get_feature_names())

	Tags_train = Tags_train.sum()
	Tags_train = Tags_train.sort_values(ascending = False)

	tags_list = Tags_train[Tags_train>20].index.to_list()

	tag_list_new = []
	for tag in tags_list:
		tag = tag.replace(" ","")
		tag_list_new.append(tag)

	for i , _tags in enumerate(questions['Tags']):
		to_drop = []
		for _i, _tag in enumerate(_tags):
			
			if _tag not in tag_list_new:
				to_drop.append(_i)
		to_drop.sort(reverse=True)
		for idx in to_drop :
			_tags.pop(idx) 

	## au bon format
	## Fonctions
	input = input.lower()
	for punctuation in string.punctuation:
		input = input.replace(punctuation, '')
	


	input = unicodedata.normalize('NFKD', input).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	

	pattern = r'[^a-zA-z.,!?/:;\"\'\s# +]' 
	input = re.sub(pattern, '', input)


	pattern = r'^\s*|\s\s*'
	input = re.sub(pattern, ' ', input).strip()

	nltk.download('wordnet')

	w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
	lemmatizer = nltk.stem.WordNetLemmatizer()

	input = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(input)]
	
	## Application fonctions
	
	nltk.download('stopwords')
	## Faire un commentaire sur le no que j'ai décidé de garder dans le rapport écrit parce qu'il ne s'agit pas de sentiment 
	## analysis
	all_stopwords = stopwords.words('english')
	
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



	## Importation du modèle
	rfc_model = pickle.load(open("deploiement_rfc_final.pkl", 'rb'))


	##prédictions
	y_pred = rfc_model.predict(tf_input)
	y_pred_inversed = multilabel_binarizer\
    .inverse_transform(y_pred)
	def Remove(tuples):
		tuples = [t for t in tuples if t]
		return tuples
	result = Remove(y_pred_inversed)
	
	return render_template('result.html',prediction = result)

if __name__ == '__main__':
	app.run(debug=True)