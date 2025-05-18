# Movie Recommendation System with Sentiment Analysis

A comprehensive movie recommendation platform that combines content-based filtering with sentiment analysis to enhance the user experience. The system provides personalized movie recommendations based on movie content and allows users to analyze the sentiment of movie reviews.

![Movie Recommendation System](https://hebbkx1anhila5yf.public.blob.vercel-storage.com/PHOTO-2025-05-18-11-53-40.jpg-6XTFTytFT0fysKCN9Shidsp9pVgmUo.jpeg)

## Features

### Movie Recommendation Engine
- **Content-Based Filtering**: Utilizes unsupervised learning techniques to recommend movies based on content similarity
- **Personalized Recommendations**: Suggests movies similar to user selections using cosine similarity
- **TMDB Integration**: Fetches movie data and posters from The Movie Database (TMDB) API
- **User Wishlist**: Allows users to save favorite movies to their personal wishlist

### Sentiment Analysis
- **Review Analysis**: Analyzes the sentiment of movie reviews using NLP techniques
- **Trained Model**: Utilizes a supervised learning model trained on 1.6 million Twitter data points
- **Real-time Feedback**: Provides immediate sentiment classification (positive/negative)

### User Management
- **Authentication System**: Secure login and registration functionality
- **User Profiles**: Personalized experience with user-specific wishlists
- **Database Integration**: SQLAlchemy ORM for efficient data management

## Technology Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework for the user interface
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms for recommendation and sentiment analysis
- **NLTK**: Natural Language Processing for text preprocessing
- **SQLAlchemy**: ORM for database operations
- **TMDB API**: External API for movie data and posters
- **Kaggle API**: For dataset acquisition


\`\`\`

## Installation

1. Clone the repository:
   \`\`\`
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   \`\`\`

2. Create and activate a virtual environment:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \`\`\`

3. Install the required packages:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Configure Kaggle API:
   - Place your `kaggle.json` file in the `~/.kaggle/` directory
   - Ensure proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

5. Run the application:
   \`\`\`
   streamlit run app.py
   \`\`\`

## Data Processing Workflow

1. **Data Acquisition**: Download TMDB movie metadata using Kaggle API
2. **Data Preprocessing**: 
   - Merge movies and credits datasets
   - Extract relevant features (genres, keywords, cast, crew, overview)
   - Clean and transform text data
3. **Feature Engineering**:
   - Create tags by combining movie features
   - Apply text preprocessing (lowercase, stemming)
4. **Vectorization**: Convert text to numerical vectors using CountVectorizer
5. **Similarity Calculation**: Compute cosine similarity between movie vectors
6. **Model Serialization**: Save processed data and similarity matrix for the web application

## Sentiment Analysis Model

The sentiment analysis component uses a supervised learning model trained on 1.6 million Twitter data points. The model:

1. Preprocesses text input using NLTK
2. Vectorizes text using TF-IDF
3. Classifies sentiment as positive or negative
4. Returns the sentiment prediction to the user

## Database Schema

The application uses SQLite with SQLAlchemy:

- **Users**: Stores user credentials and profile information
- **WishlistItems**: Maintains user-specific movie wishlists with references to TMDB movie IDs

## Future Improvements

- Implement collaborative filtering to enhance recommendation quality
- Add more advanced NLP features for deeper review analysis
- Integrate with more movie data sources
- Develop a more sophisticated UI with additional visualization options
- Add social features to allow users to share recommendations



## Acknowledgments

- TMDB for providing the movie dataset




