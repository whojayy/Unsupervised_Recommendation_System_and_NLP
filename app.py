import streamlit as st
import pickle
import pandas as pd
import requests
import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import hashlib
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationship with WishlistItem
    wishlist_items = relationship("WishlistItem", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class WishlistItem(Base):
    __tablename__ = 'wishlist_items'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    movie_id = Column(Integer, nullable=False)
    movie_title = Column(String(200), nullable=False)
    added_date = Column(String(50), nullable=False)  # Store date as string for simplicity
    
    # Relationship with User
    user = relationship("User", back_populates="wishlist_items")
    
    def __repr__(self):
        return f"<WishlistItem(movie_title='{self.movie_title}')>"

# Create database engine and session
engine = create_engine('sqlite:///movie_recommendation.db')
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Email validation function
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

# Function to fetch the poster of a movie using TMDb API
def fetch_poster(movie_id):
    try:
        # Convert to integer if it's not already
        movie_id = int(movie_id)
        
        api_key = 'e766c8844c6ffa7dcefe75500d32559a'
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            poster_path = data.get('poster_path')
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                return poster_url
            else:
                return "https://via.placeholder.com/500x750?text=No+Poster"
        else:
            return "https://via.placeholder.com/500x750?text=API+Error"
    except Exception as e:
        return "https://via.placeholder.com/500x750?text=Error"

# Function to recommend movies and fetch their posters
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]  # Recommend 6 movies

    recommend_movies = []
    recommend_posters = []
    recommend_ids = []
    
    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id  # Get the movie_id
        recommend_movies.append(movies.iloc[i[0]].title)  # Add the movie title
        recommend_posters.append(fetch_poster(movie_id))  # Fetch and add the poster URL
        recommend_ids.append(movie_id)  # Add the movie ID
        
    return recommend_movies, recommend_posters, recommend_ids

# Function to add movie to wishlist
def add_to_wishlist(user_id, movie_id, movie_title):
    if not user_id:
        return False, "User ID not found"
    
    # Convert movie_id to integer if it's not already
    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return False, f"Invalid movie ID: {movie_id}"
    
    # Print debug info to console (not visible to user)
    print(f"Adding to wishlist - User ID: {user_id}, Movie ID: {movie_id}, Title: {movie_title}")
    
    db = SessionLocal()
    try:
        # Check if movie already exists in wishlist
        existing_item = db.query(WishlistItem).filter(
            WishlistItem.user_id == user_id,
            WishlistItem.movie_id == movie_id
        ).first()
        
        if existing_item:
            db.close()
            return False, "Movie already in wishlist"
        
        # Add new wishlist item
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_item = WishlistItem(
            user_id=user_id,
            movie_id=movie_id,
            movie_title=movie_title,
            added_date=current_date
        )
        
        db.add(new_item)
        db.commit()
        db.close()
        
        # Also add to movie_wishlist table for compatibility
        add_to_movie_wishlist(user_id, movie_title)
        
        return True, "Movie added to wishlist"
    except Exception as e:
        db.rollback()
        db.close()
        # Print the full error for debugging
        import traceback
        print(f"Error adding to wishlist: {str(e)}")
        print(traceback.format_exc())
        return False, f"Error adding to wishlist: {str(e)}"
    
# Helper function to also add to movie_wishlist table
def add_to_movie_wishlist(user_id, movie_name):
    db = SessionLocal()
    try:
        # Check if already exists
        existing = db.execute(
            f"SELECT * FROM movie_wishlist WHERE user_id = {user_id} AND movie_name = '{movie_name}'"
        ).fetchone()
        
        if not existing:
            # Add to movie_wishlist table
            db.execute(
                f"INSERT INTO movie_wishlist (user_id, movie_name) VALUES ({user_id}, '{movie_name}')"
            )
            db.commit()
    except Exception as e:
        print(f"Error adding to movie_wishlist: {str(e)}")
        db.rollback()
    finally:
        db.close()    

# Function to get user's wishlist
def get_wishlist(user_id):
    if not user_id:
        st.warning("User ID not found. Please log in again.")
        return [], [], []
    
    db = SessionLocal()
    try:
        wishlist_items = db.query(WishlistItem).filter(WishlistItem.user_id == user_id).all()
        
        wishlist_movies = []
        wishlist_posters = []
        wishlist_ids = []
        
        for item in wishlist_items:
            wishlist_movies.append(item.movie_title)
            poster_url = fetch_poster(item.movie_id)
            wishlist_posters.append(poster_url)
            wishlist_ids.append(item.movie_id)
            
        return wishlist_movies, wishlist_posters, wishlist_ids
    except Exception as e:
        st.error(f"Error retrieving wishlist: {str(e)}")
        return [], [], []
    finally:
        db.close()

# Function to remove movie from wishlist
def remove_from_wishlist(user_id, movie_id):
    if not user_id:
        return False, "User ID not found"
    
    # Convert movie_id to integer if it's not already
    try:
        movie_id = int(movie_id)
    except (ValueError, TypeError):
        return False, "Invalid movie ID"
    
    db = SessionLocal()
    try:
        item = db.query(WishlistItem).filter(
            WishlistItem.user_id == user_id,
            WishlistItem.movie_id == movie_id
        ).first()
        
        if item:
            db.delete(item)
            db.commit()
            return True, "Movie removed from wishlist"
        else:
            return False, "Movie not found in wishlist"
    except Exception as e:
        db.rollback()
        return False, f"Error removing from wishlist: {str(e)}"
    finally:
        db.close()

# Function to get user ID from username
def get_user_id(username):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user:
            return user.id
        return None
    except Exception as e:
        st.error(f"Error getting user ID: {str(e)}")
        return None
    finally:
        db.close()

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        # Load the trained model and vectorizer
        model_path = 'trained_model.sav'
        vectorizer_path = 'vectorizer.sav'
        
        model = pickle.load(open(model_path, 'rb'))
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        
        # Transform the text using the vectorizer
        text_tfidf = vectorizer.transform([text])
        
        # Predict sentiment
        prediction = model.predict(text_tfidf)
        
        # Return result
        if prediction[0] == 1:
            return "Positive"
        else:
            return "Negative"
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return f"Error analyzing sentiment: {str(e)}"

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'recommended_movies' not in st.session_state:
    st.session_state.recommended_movies = []
if 'recommended_posters' not in st.session_state:
    st.session_state.recommended_posters = []
if 'recommended_ids' not in st.session_state:
    st.session_state.recommended_ids = []

# Load movie data and similarity matrix
try:
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)  
    similarity = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Movie data files not found. Please make sure the paths are correct.")
    movies = None
    similarity = None

# Sidebar for navigation
st.sidebar.title("Navigation")
if st.session_state.logged_in:
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    
    if st.sidebar.button("Home"):
        st.session_state.page = "recommendation"
        st.rerun()
        
    if st.sidebar.button("My Wishlist"):
        st.session_state.page = "wishlist"
        st.rerun()
    
    if st.sidebar.button("Sentiment Analysis"):
        st.session_state.page = "sentiment"
        st.rerun()
        
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = None
        st.session_state.page = "login"
        st.rerun()
else:
    if st.sidebar.button("Login"):
        st.session_state.page = "login"
        st.rerun()
    if st.sidebar.button("Sign Up"):
        st.session_state.page = "signup"
        st.rerun()

def login_page():
    st.title("Login to Movie Recommendation System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login", key="login_button"):
        if username and password:
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.username == username).first()
                
                if user and user.password == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_id = user.id  
                    st.session_state.page = "recommendation"
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            except Exception as e:
                st.error(f"Login error: {str(e)}")
            finally:
                db.close()
        else:
            st.warning("Please enter both username and password")

# Signup page
def signup_page():
    st.title("Sign Up for Movie Recommendation System")

    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up", key="signup_button"):
        if username and email and password and confirm_password:
            if not is_valid_email(email):
                st.error("Please enter a valid email address")
                return

            if password != confirm_password:
                st.error("Passwords do not match")
                return

            db = SessionLocal()
            try:
                existing_user = db.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()

                if existing_user:
                    st.error("Username or email already exists")
                    return

                hashed_password = hash_password(password)
                new_user = User(username=username, email=email, password=hashed_password)

                db.add(new_user)
                db.commit()
                st.success("Account created successfully! Please login.")
                st.session_state.page = "login"
                st.rerun()
            except Exception as e:
                db.rollback()
                st.error(f"Error creating account: {str(e)}")
            finally:
                db.close()
        else:
            st.warning("Please fill in all fields")

# Recommendation page
def recommendation_page():
    st.title('Movie Recommendation System')
    st.write(f"Welcome, {st.session_state.username}!")
    
    # Always refresh user_id from database
    user_id = get_user_id(st.session_state.username)
    if user_id:
        st.session_state.user_id = user_id
    else:
        st.error("User ID not found. Please log out and log in again.")
        return
    
    if movies is not None and similarity is not None:
        selected_movie_name = st.selectbox(
            'Select a movie:',
            movies['title'].values
        )
        
        if st.button('Recommend'):
            recommendations, posters, movie_ids = recommend(selected_movie_name)
            
            # Store recommendations in session state
            st.session_state.recommended_movies = recommendations
            st.session_state.recommended_posters = posters
            st.session_state.recommended_ids = movie_ids
        
        # Check if recommendations exist in session state
        if 'recommended_movies' in st.session_state and st.session_state.recommended_movies:
            # Display recommendations in a grid
            cols = st.columns(3)
            for idx, (name, poster, movie_id) in enumerate(zip(
                st.session_state.recommended_movies, 
                st.session_state.recommended_posters, 
                st.session_state.recommended_ids
            )):
                with cols[idx % 3]:
                    st.image(poster, use_container_width=True)
                    st.write(name)
                    
                    # Add to wishlist button
                    button_key = f"add_{movie_id}_{idx}"
                    if st.button("Add to Wishlist", key=button_key):
                        # Add to wishlist with proper error handling
                        try:
                            success, message = add_to_wishlist(
                                st.session_state.user_id,
                                movie_id,
                                name
                            )
                            
                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
    else:
        st.error("Movie data is not available. Please check the data files.")

# Wishlist page
def wishlist_page():
    st.title('My Wishlist')
    st.write(f"Welcome to your wishlist, {st.session_state.username}!")
    
    # Always refresh user_id from database
    user_id = get_user_id(st.session_state.username)
    if user_id:
        st.session_state.user_id = user_id
    else:
        st.error("User ID not found. Please log out and log in again.")
        return
    
    # Get user's wishlist
    wishlist_movies, wishlist_posters, wishlist_ids = get_wishlist(st.session_state.user_id)
    
    if not wishlist_movies:
        st.info("Your wishlist is empty. Add some movies from the recommendations!")
        return
    
    # Display wishlist items
    cols = st.columns(3)
    for idx, (name, poster, movie_id) in enumerate(zip(wishlist_movies, wishlist_posters, wishlist_ids)):
        with cols[idx % 3]:
            st.image(poster, use_container_width=True)
            st.write(name)
            
            button_key = f"remove_{movie_id}_{idx}"
            if st.button("Remove", key=button_key):
                success, message = remove_from_wishlist(
                    st.session_state.user_id,
                    movie_id
                )
                if success:
                    st.success(message)
                    
                    st.rerun()
                else:
                    st.error(message)

# Sentiment Analysis page
def sentiment_page():
    st.title('Sentiment Analysis')
    st.write(f"Welcome to the sentiment analyzer, {st.session_state.username}!")
    st.write("Enter your movie review and our model will analyze its sentiment and your movie experience.")
    
    # Text input for user to enter their review or text
    user_text = st.text_area("Enter your text here:", height=150)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_text:
            # Show a spinner while analyzing
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(user_text)
            
            # Display result with appropriate styling
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment} 😊")
            elif sentiment == "Negative":
                st.error(f"Sentiment: {sentiment} 😞")
            else:
                st.warning(f"Result: {sentiment}")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Add some example texts that users can try
    with st.expander("Try these examples"):
        examples = [
            "I love this movie! It's amazing and the actors did a great job.",
            "This was a terrible experience. The plot was confusing and the acting was poor.",
            "Stevan was amazing in the movie, however stacy was annoying but overall movie was great."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                # Set the text area value and analyze
                st.session_state.example_text = example
                st.rerun()
    
    # If an example was selected, populate the text area
    if 'example_text' in st.session_state:
        st.text_area("Enter your text here:", value=st.session_state.example_text, height=150, key="populated_text")
        # Clear the example text from session state
        del st.session_state.example_text

# Display the appropriate page based on the session state
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "recommendation":
    if st.session_state.logged_in:
        recommendation_page()
    else:
        st.warning("Please login to access recommendations")
        st.session_state.page = "login"
        st.rerun()
elif st.session_state.page == "wishlist":
    if st.session_state.logged_in:
        wishlist_page()
    else:
        st.warning("Please login to access your wishlist")
        st.session_state.page = "login"
        st.rerun()
elif st.session_state.page == "sentiment":
    if st.session_state.logged_in:
        sentiment_page()
    else:
        st.warning("Please login to access the sentiment analyzer")
        st.session_state.page = "login"
        st.rerun()