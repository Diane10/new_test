import streamlit as st 
import joblib,os
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")

# load Vectorizer For Gender Prediction
news_vectorizer = open("final_news_cv_vectorizer.pkl","rb")
news_cv = joblib.load(news_vectorizer)

def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model



# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def main():
	"""News Classifier"""
	st.title("News Classifier")
	# st.subheader("ML App with Streamlit")
	html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Streamlit ML App </h1>
	</div>

	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	activity = ['Prediction','NLP']
	choice = st.sidebar.selectbox("Select Activity",activity)


	if choice == 'Prediction':
		st.info("Prediction with ML")

		news_text = st.text_area("Enter News Here","Type Here")
		all_ml_models = ["LR","RFOREST","NB","DECISION_TREE"]
		model_choice = st.selectbox("Select Model",all_ml_models)

		prediction_labels = {'business': 0,'tech': 1,'sport': 2,'health': 3,'politics': 4,'entertainment': 5}
		if st.button("Classify"):
			st.text("Original Text::\n{}".format(news_text))
			vect_text = news_cv.transform([news_text]).toarray()
			if model_choice == 'LR':
				predictor = load_prediction_models("newsclassifier_Logit_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'RFOREST':
				predictor = load_prediction_models("newsclassifier_RFOREST_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'NB':
				predictor = load_prediction_models("newsclassifier_NB_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'DECISION_TREE':
				predictor = load_prediction_models("newsclassifier_CART_model.pkl")
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_key(prediction,prediction_labels)
			st.success("News Categorized as:: {}".format(final_result))



	st.sidebar.subheader("About")




if __name__ == '__main__':
	main()

