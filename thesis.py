import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
from pymongo import MongoClient
import bcrypt
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('label_encoders.pkl','rb'))

connection_string = "mongodb+srv://cuetsys:HCcROtZKCvESrEna@reviewdatabase.k4gemlk.mongodb.net/?retryWrites=true&w=majority&appName=reviewDatabase"
client = MongoClient(connection_string)
db = client["thesis-user"]
users_collection = db["thesis"]
qa_collection = db["qa"]
Data = db['data']
blogs_collection = db['blogs']
# Create a dummy dataset for blogs
blogs = pd.DataFrame({
    'title': ['Blog 1', 'Blog 2', 'Blog 3'],
    'content': ['This is blog 1', 'This is blog 2', 'This is blog 3']
})

# Create a dummy dataset for Q&A
qa_data = pd.DataFrame({
    'question': ['What is Streamlit?', 'How do I use Streamlit?', 'What is a data scientist?'],
    'answer': ['Streamlit is an open-source app framework for Machine Learning and Data Science.', 'You can use Streamlit to create interactive web apps with Python.', 'A data scientist is a professional who analyzes and interprets complex data sets.']
})


# Signup functionality
def signup():
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            # Check if the username already exists
            existing_user = users_collection.find_one({"username": username})
            if existing_user:
                st.error("Username already exists")
            else:
                # Hash the password
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

                # Create a new user
                new_user = {"username": username, "password": hashed_password, "ac": False}
                users_collection.insert_one(new_user)
                st.success("Sign up successful!")

# Signin functionality
def fun():
    st.title("Model Prediction")
    name = st.text_input("Policy Holder Name")
    policy_no = st.text_input("Policy No.")
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender Option : (Male : M , Female : F)", ["M", "F"])
    job = st.text_input("Occupation")
    address = st.selectbox("Division (Ex: Dhaka , Chittagong)",
                           ["Dhaka", "Rajshahi", "Khulna", "Sylhet", "Chittagong", "Barishal", "Mymenshingh",
                            "Rangpur"])
    policy_age_years = st.number_input("Policy Age (Years)", min_value=0)
    coverage_type = st.selectbox("Coverage Type", ["Fire", "Theft", "Water Damage", "Storm", "Riot", "Civil Commotion"])
    coverage_amount = st.number_input("Coverage Amount", min_value=0)
    previous_claims = st.number_input("Previous Claims", min_value=0)
    claim_date = st.date_input("Claim Date")
    incident_date = st.date_input("Incident Date")
    claim_type = st.selectbox("Claim Type",
                              ["Fire", "Theft", "Water Damage", "Storm", "Riot", "Civil Commotion"])
    claim_amount = st.number_input("Claim Amount", min_value=0)
    claim_severity = st.selectbox("Claim Severity", ["Major", "Minor"])
    options = ["Yes", "No"]
    geographic_location = st.selectbox("Geographic Location", ['Low-risk', 'High-risk', 'Moderate-risk'])
    suspicious_circumstances = st.selectbox("Suspicious Circumstances", options)
    financial_difficulties = st.selectbox("Financial Difficulties", options)
    exaggerated_damages = st.selectbox("Exaggerated Damages", options)
    discrepancies = st.selectbox("Discrepancies", options)
    delayed_reporting = st.selectbox("Delayed Reporting", options)
    witness_statements = st.selectbox("Witness Statements", options)
    surveillance_evidence = st.selectbox("Surveillance Evidence", options)

    claim_date = datetime.combine(claim_date, datetime.min.time())
    incident_date = datetime.combine(incident_date, datetime.min.time())
    # Collecting data into a dictionary
    valid = 0
    data = {
        "Name": [name],
        "Policy No.": [policy_no],
        "Age": [age],
        "Gender": [gender],
        "Occupation": [job],
        "Address": [address],
        "Policy Age (Years)": [policy_age_years],
        "Coverage type": [coverage_type],
        "Coverage Amount": [coverage_amount],
        "Previous Claims": [previous_claims],
        "Claim Frequency": [policy_age_years / previous_claims if previous_claims != 0 else 0.0],
        "Claim Date": [claim_date],
        "Incedent Date": [incident_date],
        "Claim Type": [claim_type],
        "Claim Amount": [claim_amount],
        "Claim Severity": [claim_severity],
        "Geographic Location": [geographic_location],
        "Suspicious Circumstances": [suspicious_circumstances],
        "Financial Difficulties": [financial_difficulties],
        "Exaggerated Damages": [exaggerated_damages],
        "Discrepancies": [discrepancies],
        "Delayed Reporting": [delayed_reporting],
        "Witness Statements": [witness_statements],
        "Surveillance Evidence": [surveillance_evidence]
    }
    df = pd.DataFrame(data)
    df.drop(['Claim Date', 'Incedent Date'], axis=1, inplace=True)

    st.write("Here is the data you entered:")
    st.dataframe(df)
    columns_to_extract = [
        'Suspicious Circumstances', 'Financial Difficulties', 'Exaggerated Damages',
        'Discrepancies', 'Claim Frequency', 'Age', 'Address', 'Claim Amount',
        'Policy Age (Years)', 'Surveillance Evidence']
    if st.button("Predict"):
        for col in df.select_dtypes(include=['object']).columns:
            encoder.fit(df[col].unique())
            df[col] = encoder.transform(df[col])
        extracted_df = df[columns_to_extract]
        scaled_data = scaler.transform(extracted_df)
        predictions = model.predict(scaled_data)
        prediction_prob = model.predict_proba(extracted_df)[:, 1]
        if predictions[0] == 0:
            st.markdown(
                f"<h3 style='color: red;'>Alert: Potential Fraud Detected! (Probability: {prediction_prob[0]:.2f})</h3>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<h3 style='color: green;'>No Fraud Detected (Probability: {prediction_prob[0]:.2f})</h3>",
                unsafe_allow_html=True)
        Data.insert_one(data)
        Data.insert_one(df.to_dict('records')[0])  # Uncomment this line after setting up MongoDB
        st.success("Data saved successfully!")
def signin():
    st.title("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        # Find the user in the database
        user = users_collection.find_one({"username": username})
        if user:
            # Check if the password matches
            if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
                st.success("Sign in successful!")
                # Store the user object in the session state
                st.session_state.user = user
            else:
                st.error("Incorrect password")
        else:
            st.error("User not found")

def ask_question():
    st.title("Ask a Question")
    question = st.text_area("Enter your question")
    if st.button("Submit Question"):
        if question:
            qa_data = {"question": question, "answers": []}
            qa_collection.insert_one(qa_data)
            st.success("Question submitted successfully!")
        else:
            st.warning("Please enter a question.")

def answer_question():
    user = st.session_state.get('user')
    if user and user.get("ac", False):
        st.title("Answer a Question")
        questions = list(qa_collection.find({}, {"question": 1, "_id": 0}))
        selected_question = st.selectbox("Select a question", [q["question"] for q in questions])
        answer = st.text_area("Enter your answer")
        if st.button("Submit Answer"):
            if answer:
                qa_data = qa_collection.find_one({"question": selected_question})
                if len(qa_data["answers"]) < 5:
                    qa_collection.update_one(
                        {"question": selected_question},
                        {"$push": {"answers": answer}}
                    )
                    st.success("Answer submitted successfully!")
                else:
                    st.warning("Maximum number of answers reached for this question.")
            else:
                st.warning("Please enter an answer.")
    else:
        st.warning("You do not have access to answer questions.")

def view_qa():
    st.title("Q&A")
    questions = list(qa_collection.find({}, {"question": 1, "answers": 1, "_id": 0}))
    for qa in questions:
        st.subheader(qa["question"])
        for answer in qa["answers"]:
            st.write(answer)
# Create the app
def add_blog():
    user = st.session_state.get('user')
    if user and user.get("ac", False):
        st.title("Add New Blog")
        blog_title = st.text_input("Blog Title")
        blog_content = st.text_area("Blog Content")
        author_name = st.text_input("Author Name", value=user["username"])
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_encoded = base64.b64encode(image_bytes).decode()
        if st.button("Submit"):
            if blog_title and blog_content:
                blog_data = {
                    "title": blog_title,
                    "content": blog_content,
                    "author": author_name,
                    "image": image_encoded if uploaded_file is not None else None
                }
                blogs_collection.insert_one(blog_data)
                st.success("Blog added successfully!")
            else:
                st.warning("Please enter a title and content for the blog.")
    else:
        st.warning("You do not have access to add new blogs.")

def view_blogs():
    st.title("Home")
    blogs = blogs_collection.find({}, {"_id": 0})
    for blog in blogs:
        st.subheader(blog["title"])
        st.write(f"Author: {blog['author']}")
        if blog.get("image"):
            image_bytes = base64.b64decode(blog["image"])
            image = Image.open(BytesIO(image_bytes))
            st.image(image, caption="Blog Image", use_column_width=True)
        st.write(blog["content"])
        st.write("---")

def main():
    st.set_page_config(page_title="My App", page_icon=":guardsman:", layout="wide")

    # Create the sidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Sign In", "Sign Up", "Home", "Model Prediction", "Q&A"],
            icons=["person-fill", "pencil-fill", "house-fill", "bar-chart-fill", "question-circle-fill"],
            default_index=0,
        )

    # Sign in page
    if selected == "Sign In":
        signin()

    # Sign up page
    elif selected == "Sign Up":
        signup()

    # Home page
    elif selected == "Home":
        home_option = option_menu(
            menu_title="Home",
            options=["View Blogs", "Add New Blog"],
            icons=["eye-fill", "pencil-fill"],
            default_index=0,
        )
        if home_option == "View Blogs":
            view_blogs()
        elif home_option == "Add New Blog":
            add_blog()

    # Model Prediction page
    elif selected == "Model Prediction":
        if st.session_state.get('user') and st.session_state.user.get("ac", True):
            fun()
        else:
            st.warning("You do not have access to this page.")
    # Q&A page
    elif selected == "Q&A":
        qa_option = option_menu(
            menu_title="Q&A",
            options=["View Q&A", "Ask Question", "Answer Question"],
            icons=["eye-fill", "question-circle-fill", "pencil-fill"],
            default_index=0,
        )
        if qa_option == "View Q&A":
            view_qa()
        elif qa_option == "Ask Question":
            ask_question()
        elif qa_option == "Answer Question":
            answer_question()

if __name__ == "__main__":
    main()