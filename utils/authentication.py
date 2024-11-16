# app.py

import streamlit as st
import sqlite3
import uuid
import time
import re  # For email validation
from streamlit_authenticator import Hasher

# ===============================
# Streamlit Page Configuration
# ===============================
st.set_page_config(page_title="🔐 Secure Login System", page_icon="🔐", layout="centered")

# ===============================
# Database Setup
# ===============================
def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect("users.db")
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
    return conn

def initialize_db():
    """Initialize the users table in the database."""
    conn = create_connection()
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                security_question TEXT NOT NULL,
                hashed_security_answer TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()
    else:
        st.error("Failed to initialize the database.")

# Initialize the database
initialize_db()

# ===============================
# Session State Initialization
# ===============================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ===============================
# Utility Functions
# ===============================
def is_valid_email(email: str) -> bool:
    """Validate the email format using regex."""
    email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(email_regex, email) is not None

# ===============================
# User Registration Function
# ===============================
def register_user(username: str, email: str, password: str, security_question: str, security_answer: str) -> bool:
    """
    Registers a new user with hashed password and security answer.
    """
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return False

    cursor = conn.cursor()
    # Check if username or email already exists
    cursor.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
    if cursor.fetchone():
        st.error("Username or email already exists. Please choose a different one.")
        conn.close()
        return False

    # Hash the password and security answer
    hashed_password = Hasher.hash(password)
    hashed_security_answer = Hasher.hash(security_answer)

    # Insert the new user into the database
    try:
        cursor.execute("""
            INSERT INTO users (username, email, hashed_password, security_question, hashed_security_answer)
            VALUES (?, ?, ?, ?, ?)
        """, (username, email, hashed_password, security_question, hashed_security_answer))
        conn.commit()
        st.success("Registration successful! You can now log in.")
        success = True
    except sqlite3.Error as e:
        st.error(f"Registration failed: {e}")
        success = False
    finally:
        conn.close()
    return success

# ===============================
# User Login Function
# ===============================
def login_user(username: str, password: str) -> bool:
    """
    Logs in a user by verifying their credentials.
    """
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return False

    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        hashed_password = result[0]
        if Hasher.check_pw(password, hashed_password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            return True
        else:
            st.error("Incorrect password.")
            return False
    else:
        st.error("Username does not exist.")
        return False

# ===============================
# Password Reset Functions
# ===============================
def get_security_question(username: str):
    """
    Retrieves the security question for the given username.
    """
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return None

    cursor = conn.cursor()
    cursor.execute("SELECT security_question FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        st.error("Username does not exist.")
        return None

def verify_security_answer(username: str, answer: str) -> bool:
    """
    Verifies the security answer for the given username.
    """
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return False

    cursor = conn.cursor()
    cursor.execute("SELECT hashed_security_answer FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        hashed_security_answer = result[0]
        return Hasher.check_pw(answer, hashed_security_answer)
    else:
        st.error("Username does not exist.")
        return False

def reset_password(username: str, new_password: str) -> bool:
    """
    Resets the user's password to the new hashed password.
    """
    conn = create_connection()
    if conn is None:
        st.error("Database connection failed.")
        return False

    cursor = conn.cursor()
    hashed_password = Hasher.hash(new_password)
    try:
        cursor.execute("UPDATE users SET hashed_password = ? WHERE username = ?", (hashed_password, username))
        conn.commit()
        st.success("Password has been reset successfully! You can now log in with your new password.")
        success = True
    except sqlite3.Error as e:
        st.error(f"Password reset failed: {e}")
        success = False
    finally:
        conn.close()
    return success

# ===============================
# Protected Content Function
# ===============================
def show_protected_content():
    """
    Displays content for logged-in users.
    """
    st.title(f"Welcome, {st.session_state.username}!")
    st.write("🔒 **This is protected content only visible to logged-in users.**")
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("You have been logged out.")

# ===============================
# Registration Form
# ===============================
def registration_form():
    """
    Renders the registration form.
    """
    st.header("📝 Register")
    with st.form("register_form"):
        username = st.text_input("Username", max_chars=50)
        email = st.text_input("Email")  # Removed type="email"
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        security_question = st.selectbox(
            "Select a Security Question",
            [
                "What was your childhood nickname?",
                "What is the name of your favorite childhood friend?",
                "What was the name of your first pet?",
                "What was the first concert you attended?"
            ]
        )
        security_answer = st.text_input("Security Answer", type="password")
        submitted = st.form_submit_button("Register")

    if submitted:
        if not username or not email or not password or not confirm_password or not security_answer:
            st.error("Please fill out all fields.")
        elif not is_valid_email(email):
            st.error("Please enter a valid email address.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            register_user(username, email, password, security_question, security_answer)

# ===============================
# Login Form
# ===============================
def login_form():
    """
    Renders the login form.
    """
    st.header("🔑 Login")
    with st.form("login_form"):
        username = st.text_input("Username", max_chars=50)
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if not username or not password:
            st.error("Please fill out all fields.")
        else:
            login_success = login_user(username, password)
            if login_success:
                st.experimental_rerun()

# ===============================
# Password Reset Request Form
# ===============================
def password_reset_request_form():
    """
    Renders the password reset request form.
    """
    st.header("🔒 Forgot Password")
    with st.form("password_reset_request_form"):
        username = st.text_input("Username")
        submitted = st.form_submit_button("Request Password Reset")

    if submitted:
        if not username:
            st.error("Please enter your username.")
        else:
            security_question = get_security_question(username)
            if security_question:
                st.subheader("Security Question")
                answer = st.text_input(security_question, type="password")
                if st.button("Submit Answer"):
                    if verify_security_answer(username, answer):
                        new_password = st.text_input("New Password", type="password")
                        confirm_new_password = st.text_input("Confirm New Password", type="password")
                        if st.button("Reset Password"):
                            if not new_password or not confirm_new_password:
                                st.error("Please fill out all password fields.")
                            elif new_password != confirm_new_password:
                                st.error("Passwords do not match.")
                            else:
                                reset_success = reset_password(username, new_password)
                                if reset_success:
                                    st.success("Password reset successful! You can now log in with your new password.")
                    else:
                        st.error("Incorrect security answer.")

# ===============================
# Main Application Logic
# ===============================
def main():
    """
    Main function to run the Streamlit app.
    """
    if st.session_state.logged_in:
        show_protected_content()
    else:
        st.title("🔐 Secure Login System")
        menu = ["Login", "Register", "Forgot Password"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            login_form()
        elif choice == "Register":
            registration_form()
        elif choice == "Forgot Password":
            password_reset_request_form()

if __name__ == "__main__":
    main()

