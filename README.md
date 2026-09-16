-> Crop Yield Prediction

A machine learning web app that predicts crop yield based on crop type, season, state, rainfall, and farming inputs. Built with Streamlit and scikit-learn, trained on historical Indian crop data.

🔗 Live demo: https://crop-yield-prediction-4ezemq9mw2fkchsasa5bys.streamlit.app/


Tech Stack
Python
Streamlit – web app UI
scikit-learn – Decision Tree Regressor model
pandas / numpy – data handling
joblib – model serialization


Project Structure
├── app.py                     # Streamlit app
├── decision_tree_model.pkl    # Trained model
├── model_columns.pkl          # Feature column order used at training time
├── categories.pkl             # Valid Crop / Season / State values for the UI
├── requirements.txt           # Python dependencies
├── First_Model.ipynb          # Model training notebook
├── EDA_IMPL_DATA.ipynb        # Exploratory data analysis
└── transformed_crop_yield.csv # Training dataset
How It Works
Historical crop data is cleaned and transformed (log-transforms applied to skewed numeric columns like Area, Fertilizer, Pesticide, and the target Yield).
Categorical features (Crop, Season, State) are one-hot encoded.
A DecisionTreeRegressor is trained to predict Yield_log.
The trained model, the exact training column order, and the list of valid category values are saved as .pkl files.
The Streamlit app loads these files, builds a matching input row from user selections, and reverses the log transform to show the final predicted yield.
Running Locally
bash
git clone https://github.com/ricky231b263/crop-yield-prediction.git
cd crop-yield-prediction
pip install -r requirements.txt
streamlit run app.py
