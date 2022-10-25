# Disaster Response Pipeline Project

## Table of Contents

1. [Motivation](#motivation)
2. [Setup](#setup)
3. [Repository Structure](#structure)
4. [Usage](#usage)
5. [License and Acknowledgements](#license)


### Motivation <a name="motivation"></a>
This is the second project assignment of the Udacity course 'Data Scientist'.
Using a dataset of labeled messages a model ist trained to classify messages into the 36 categories provided. Additionaly some plots inform about the training dataset.


### Setup <a name="setup"></a>
1. Run the following commands in the project's root directory to set up database and model:

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app directory to run the flask web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Repository Structure <a name="structure"></a>
- this README file
- app: contains the .html templates as well as the python code for running the flask web app
- data: contains the .csv files for messages and categories, which the model is trained and evaluated on. Also contains the python code for loading and cleaning this data for further steps.
- model: contains the python code to train a model and save it as pickle file.
- .gitignore file to keep the repo clean
- other python repo scaffolding

### Usage <a name="usage"></a>
Browse to the website (http://0.0.0.0:3001/, if running from local), type a message in English and hit the button 'Classify Message'. The categories which are attributed to the message are highlighted.

![landingpage](/docs/landingpage.png?raw=true "Landingpage")

![resultpage](/docs/resultpage.png?raw=true "Landingpage")

### License and Acknowledgements <a name="license"></a>
The data which is used in this project was provided by [Figure Eight (now Appen)](https://appen.com/).

Feel free to use the code in any way you like to.
