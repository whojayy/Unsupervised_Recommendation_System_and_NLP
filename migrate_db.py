import sqlite3
from sqlalchemy import create_engine, text
from app import Base  # Import the Base from your main app

# Create database engine
engine = create_engine('sqlite:///movie_recommendation.db')

def add_poster_path_column():
    # Connect to SQLite database
    conn = sqlite3.connect('movie_recommendation.db')
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(wishlist_items)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'poster_path' not in columns:
            # Add the new column
            cursor.execute('''
                ALTER TABLE wishlist_items
                ADD COLUMN poster_path TEXT
            ''')
            print("Added poster_path column successfully")
        else:
            print("poster_path column already exists")
            
        conn.commit()
    except Exception as e:
        print(f"Error: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    # Add the new column
    add_poster_path_column()
    
    # Recreate all tables that don't exist
    Base.metadata.create_all(engine)
    print("Database migration completed")