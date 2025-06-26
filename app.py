#app.py
from flask import Flask, render_template, request, redirect, session, flash, url_for, send_from_directory
import pymysql
import pandas as pd
import os
from werkzeug.utils import secure_filename
from datetime import datetime , timedelta
import hashlib
import logging
from collections import defaultdict
from flask import jsonify
import json
import openai
import subprocess
import secrets
import sys 
from weasyprint.text.ffi import pango
from flask import make_response
import ctypes.util

from flask_socketio import SocketIO, join_room, leave_room, send
eye_tracker_process = None

CODING_KEYWORDS = [
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "swift", "kotlin", "golang",
    "algorithm", "data structure", "coding", "programming", "function", "method", "loop", "iteration",
    "variable", "array", "list", "dictionary", "object", "class", "inheritance", "polymorphism",
    "recursion", "pointer", "memory management", "binary tree", "graph", "sorting", "searching",
    "html", "css", "sql", "database", "nosql", "query", "server", "api", "rest", "http", "json", "xml",
    "debug", "debugger", "compile", "interpreter", "runtime", "syntax", "ide", "sdk", "framework", 
    "library", "module", "package", "version control", "git", "command line", "terminal",
    "frontend", "backend", "fullstack", "devops", "agile", "scrum", "testing", "unittest",
    "exception", "error handling", "async", "await", "thread", "process", "network", "socket",
    "security", "encryption", "authentication", "authorization", "machine learning", "ai",
    "neural network", "deep learning", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"
]


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # For PDF embeddings
from langchain_huggingface import HuggingFacePipeline 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

import shutil 

app = Flask(__name__) 

from flask_socketio import SocketIO
socketio = SocketIO(app, async_mode='eventlet')



app.secret_key = 'your-secret-key'  
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Define a path for storing vector stores
VECTOR_STORE_BASE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'vector_stores')
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


local_llm_pipeline = None
try:
    logger.info("Attempting to load local LLM: google/flan-t5-small...")
    local_llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-small", # Using a smaller, free model
        task="text2text-generation",
        pipeline_kwargs={
            "max_new_tokens": 150,
            "do_sample": True, 
            "temperature": 0.7,
            "top_p": 0.95,
        }
        
    )
    logger.info("Successfully loaded local LLM: google/flan-t5-small.")
except Exception as e_global_llm:
    logger.error(f"Failed to load HuggingFace LLM (google/flan-t5-small) globally: {e_global_llm}", exc_info=True)
    local_llm_pipeline = None 

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
   
    logger.info(f"Attempting to serve: {filename} from {app.config['UPLOAD_FOLDER']}") 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)


def get_db_connection():
    """Create and return a database connection"""
    try:
        return pymysql.connect(
            host='localhost',
            user='root',
            password='Neha@2101',
            database='smart_tutoring',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def hash_password(password):
    """Create a SHA-256 hash of the password"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_incomplete_profile(user):
    """Check if user profile is incomplete based on role"""
    if user['role'] == 'student':
        return (
            not user.get('roll_number') or 
            not user.get('department') or 
            not user.get('semester')
        )
    elif user['role'] == 'teacher':
        return (
            not user.get('department') or 
            not user.get('semesters_taught') or 
            not user.get('subjects_per_semester')
        )
    return False

@app.route('/')
def home():
    """Redirect to login page"""
    return redirect('/login')

@app.route('/get-subjects', methods=['GET'])
def get_subjects():
    """Get subjects for a specific semester and department"""
    semester = request.args.get('semester')
    department = request.args.get('department')

    if not semester or not department:
        return {'error': 'Semester and Department are required'}, 400

    try:
        # Get absolute path to Excel file
        excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
        
        if not os.path.exists(excel_path):
            logger.error(f"Excel file not found at: {excel_path}")
            return {'error': f'Excel file not found at: {excel_path}'}, 500

        df = pd.read_excel(excel_path)
        
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Check columns
        required_columns = ['Semester', 'Department', 'Subject']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f'Column "{col}" not found in Excel. Available columns: {df.columns.tolist()}')
                return {'error': f'Column "{col}" not found in Excel'}, 500

        # Normalize semester values for comparison
        def normalize_sem(s):
            s = str(s).lower().replace('semester', '').strip()
            if s in ['1', '1st']: return '1'
            if s in ['2', '2nd']: return '2'
            # Add other cases as needed
            return s

        # Filter
        filtered = df[
            (df['Semester'].apply(normalize_sem) == normalize_sem(semester)) &
            (df['Department'].str.strip().str.lower() == department.lower().strip())
        ]

        subjects = filtered['Subject'].tolist()
        return {'subjects': subjects}

    except Exception as e:
        logger.error(f"Error in get_subjects: {str(e)}")
        return {'error': str(e)}, 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        enable_eye_tracking_on_login = request.form.get('enable_eye_tracking') == 'yes' # Check the checkbox

        hashed_password = hash_password(password)
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE username=%s AND password=%s AND role=%s",
                    (username, hashed_password, role)
                )
                user = cur.fetchone()
            conn.close()

            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['status'] = user['status'] 
                session['eye_control_active'] = False # This is for the SERVER-SIDE script state

                if user.get('department'):
                    session['department'] = user['department']
                if user.get('semester'):
                    session['semester'] = user['semester']

                # Eye Tracking Preference
                if enable_eye_tracking_on_login:
                    session['initiate_web_eye_tracking_on_load'] = True # THIS flag redirects to the WEB tracker
                    session['eye_control_enabled_preference'] = True 
                    logger.info(f"User {username} opted to enable WEB eye tracking on login. Will redirect to /web_eye_tracker.")
                else:
                    session['initiate_web_eye_tracking_on_load'] = False
                    session['eye_control_enabled_preference'] = False
            
                    session['eye_control_active'] = False # This refers to the server-side script state

                if user['role'] == 'teacher' and user['status'] == 'pending':
                    flash('Your account is pending approval. Please wait.', 'info')
                    return redirect('/dashboard')

                if user['status'] == 'approved' and is_incomplete_profile(user):
                    if role == 'student':
                        return redirect('/complete-profile')
                    elif role == 'teacher':
                        return redirect('/complete-teacher-profile')
                
                flash('Login successful!', 'success')
                return redirect('/dashboard')
            else:
                error = 'Invalid username, password, or role'
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            error = 'An error occurred. Please try again.'

    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration"""
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirmPassword']
        role = request.form['role']

        if password != confirm:
            error = 'Passwords do not match'
        else:
            try:
                
                hashed_password = hash_password(password)
                
                conn = get_db_connection()
                with conn.cursor() as cur:
                    
                    cur.execute("SELECT id FROM users WHERE username=%s OR email=%s", 
                              (username, email))
                    
                    if cur.fetchone():
                        error = 'Username or email already exists'
                    else:
                        user_status = 'approved'
                        if role == 'teacher':
                            
                            semesters = request.form.getlist('semesters')
                            subjects = request.form.getlist('teaching_subjects')
                
                            
                            subjects_str = ','.join(subjects)
                            semesters_str = ','.join(semesters)

                            user_status = 'pending'
                
                            cur.execute("""
                                INSERT INTO users (username, email, password, role, 
                                semesters_taught, subjects_per_semester)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """, (username, email, hashed_password, role, semesters_str, subjects_str))
                        else:
                            cur.execute("""
                            INSERT INTO users (username, email, password, role, status)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (username, email, hashed_password, role, user_status))
                        
                        conn.commit()
                        flash('Account created successfully! Please login.', 'success')
                        return redirect('/login')
                conn.close()
            except Exception as e:
                logger.error(f"Signup error: {str(e)}")
                error = f'An error occurred during signup: {str(e)}'
    
    return render_template('signup.html', error=error)


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

    username = session['username']
    user_id = session['user_id']
    role = session['role']
    status = session.get('status', 'approved')

    if role == 'teacher' and status == 'pending':
        return render_template('dashboard.html', username=username, role=role, user_status='pending')

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user_details = cursor.fetchone()
        if not user_details:
            flash('User details not found. Please login again.', 'error')
            session.clear()
            return redirect(url_for('login'))

        # Variables for student view
        grouped_assignments = defaultdict(list)
        grouped_materials = defaultdict(list)
        pending_tests_for_dashboard = []
        active_lectures_for_student = []
        student_level = None
        roll, student_department, student_semester, subjects_list_for_student = None, None, None, []


        # Variables for teacher view
        teacher_assignments = []
        teacher_materials = []
        semesters_taught_list = []
        subjects_per_semester_data = {}
        teacher_semester_details = defaultdict(lambda: {'students': [], 'materials': [], 'assignments': [], 'tests_with_results': []})
        teacher_active_lectures = []

        # Variables for admin view
        teachers_list = []
        pending_teachers_list = []


        if role == 'student':
            roll = user_details.get('roll_number')
            student_department = user_details.get('department')
            student_semester = user_details.get('semester')
            subjects_str = user_details.get('subjects', '')
            subjects_list_for_student = [s.strip() for s in subjects_str.split(',') if s.strip()] if subjects_str else []
            student_level = user_details.get('level')

            logger.info(f"Student Dashboard: User: {username}, Sem: {student_semester}, Subjects: {subjects_list_for_student}, Level: {student_level}")

            if student_semester and subjects_list_for_student:
                placeholders = ','.join(['%s'] * len(subjects_list_for_student))
                base_params = [student_semester] + subjects_list_for_student
                difficulty_filter_sql = ""
                difficulty_params = []

                if student_level:
                    difficulty_filter_sql = " AND (a.difficulty_level = %s OR a.difficulty_level = 'General' OR a.difficulty_level IS NULL OR a.difficulty_level = '')" # Alias 'a' for assignments
                    difficulty_params.append(student_level)
                else: 
                    difficulty_filter_sql = " AND (a.difficulty_level = 'Novice' OR a.difficulty_level = 'General' OR a.difficulty_level IS NULL OR a.difficulty_level = '')"

                assignments_query = f"""
                    SELECT a.*, u.username as teacher_name 
                    FROM assignments a
                    JOIN users u ON a.uploaded_by = u.username
                    WHERE a.semester = %s AND a.subject IN ({placeholders}) {difficulty_filter_sql.replace('a.difficulty_level','a.difficulty_level')} 
                    ORDER BY a.upload_date DESC
                """
                final_assignments_params = base_params + difficulty_params
                cursor.execute(assignments_query, final_assignments_params)
                student_assignments_data = cursor.fetchall()
                for item in student_assignments_data:
                    item['file_url'] = url_for('serve_upload', filename=item['filename'])
                    grouped_assignments[item['subject']].append(item)
                
                # Adjust difficulty_filter_sql for materials (alias 'm')
                if student_level:
                    difficulty_filter_sql_mat = " AND (m.difficulty_level = %s OR m.difficulty_level = 'General' OR m.difficulty_level IS NULL OR m.difficulty_level = '')"
                else:
                    difficulty_filter_sql_mat = " AND (m.difficulty_level = 'Novice' OR m.difficulty_level = 'General' OR m.difficulty_level IS NULL OR m.difficulty_level = '')"

                materials_query = f"""
                    SELECT m.*, u.username as teacher_name 
                    FROM materials m
                    JOIN users u ON m.teacher = u.username
                    WHERE m.semester = %s AND m.subject IN ({placeholders}) {difficulty_filter_sql_mat}
                    ORDER BY m.upload_date DESC
                """
                final_materials_params = base_params + difficulty_params # difficulty_params is same if student_level is used
                cursor.execute(materials_query, final_materials_params)
                student_materials_data = cursor.fetchall()
                for item in student_materials_data:
                    item['file_url'] = url_for('serve_upload', filename=item['filename'])
                    grouped_materials[item['subject']].append(item)

                # Adjust difficulty_filter_sql for tests (alias 't')
                if student_level:
                    difficulty_filter_sql_test = " AND (t.difficulty_level = %s OR t.difficulty_level = 'General' OR t.difficulty_level IS NULL OR t.difficulty_level = '')"
                else:
                    difficulty_filter_sql_test = " AND (t.difficulty_level = 'Novice' OR t.difficulty_level = 'General' OR t.difficulty_level IS NULL OR t.difficulty_level = '')"
                
                tests_query = f"""
                    SELECT t.*, t.teacher_username as teacher_name 
                    FROM tests t
                    WHERE t.semester = %s AND t.subject IN ({placeholders}) {difficulty_filter_sql_test}
                    ORDER BY t.due_date ASC
                """
                final_tests_params = base_params + difficulty_params # difficulty_params is same
                cursor.execute(tests_query, final_tests_params)
                all_subject_tests = cursor.fetchall()

                if all_subject_tests:
                    for test_item in all_subject_tests:
                        cursor.execute("SELECT id FROM test_attempts WHERE student_id = %s AND test_id = %s", (user_id, test_item['id']))
                        attempt = cursor.fetchone()
                        if not attempt: 
                            pending_tests_for_dashboard.append(test_item)
            
                if subjects_list_for_student:
                    live_lectures_query = f"""
                        SELECT id, teacher_id, teacher_username, subject, semester, title, meeting_link, status, started_at 
                        FROM live_lectures 
                        WHERE status = 'live' AND semester = %s AND subject IN ({",".join(["%s"]*len(subjects_list_for_student))})
                        ORDER BY started_at DESC
                    """
                    cursor.execute(live_lectures_query, [student_semester] + subjects_list_for_student)
                    active_lectures_for_student = cursor.fetchall()

        elif role == 'teacher':
            logger.info(f"--- TEACHER DASHBOARD FOR: {username} (User ID: {user_id}) ---")
            
            semesters_taught_str = user_details.get('semesters_taught', '')
            if semesters_taught_str:
                semesters_taught_list = sorted(list(set(s.strip() for s in semesters_taught_str.split(',') if s.strip())))
            
            raw_sps = user_details.get('subjects_per_semester', '')
            logger.info(f"Raw 'subjects_per_semester' from DB: '{raw_sps}'")
            
            if raw_sps:
                try:
                    if raw_sps.startswith('{') and raw_sps.endswith('}'):
                        processed_sps_json_str = raw_sps.replace("'", '"')
                        try:
                            parsed_json = json.loads(processed_sps_json_str)
                            for sem_key_json, subs_json in parsed_json.items():
                                sem_key_clean = sem_key_json.strip()
                                if isinstance(subs_json, str): subjects_per_semester_data[sem_key_clean] = [s.strip() for s in subs_json.split(',') if s.strip()]
                                elif isinstance(subs_json, list): subjects_per_semester_data[sem_key_clean] = [str(s).strip() for s in subs_json if str(s).strip()]
                                else: subjects_per_semester_data[sem_key_clean] = []
                        except json.JSONDecodeError:
                            logger.warning(f"JSONDecodeError for subjects_per_semester: '{raw_sps}'. Falling back.")
                            subjects_per_semester_data = {}
                    
                    if not subjects_per_semester_data and ';' in raw_sps:
                        for block in raw_sps.split(';'):
                            block = block.strip()
                            if ':' in block:
                                sem, subs_str = block.split(':', 1); sem_key = sem.strip()
                                if sem_key: subjects_per_semester_data[sem_key] = [s.strip() for s in subs_str.split(',') if s.strip()]
                    elif not subjects_per_semester_data and ':' in raw_sps:
                        sem, subs_str = raw_sps.split(':', 1); sem_key = sem.strip()
                        if sem_key: subjects_per_semester_data[sem_key] = [s.strip() for s in subs_str.split(',') if s.strip()]
                    elif not subjects_per_semester_data and semesters_taught_list and len(semesters_taught_list) == 1:
                        subjects_per_semester_data[semesters_taught_list[0]] = [s.strip() for s in raw_sps.split(',') if s.strip()]
                    elif not subjects_per_semester_data and raw_sps:
                        logger.warning(f"Could not parse 'subjects_per_semester' ('{raw_sps}') for teacher {username}.")
                except Exception as e_parse_sps:
                    logger.error(f"Exception parsing subjects_per_semester for {username}: {e_parse_sps}", exc_info=True)
                    subjects_per_semester_data = {}
            logger.info(f"Final Parsed 'subjects_per_semester_data': {subjects_per_semester_data}")

            try:
                cursor.execute("SELECT id, title, subject, semester, started_at, meeting_link FROM live_lectures WHERE teacher_id = %s AND status = 'live' ORDER BY started_at DESC", (user_id,))
                teacher_active_lectures = cursor.fetchall()
            except Exception as e_tal: logger.error(f"Error fetching teacher's active lectures: {e_tal}")

            for sem_taught_key in semesters_taught_list:
                current_sem_data = teacher_semester_details[sem_taught_key]
                subjects_for_this_sem_key = subjects_per_semester_data.get(sem_taught_key, [])
                
                if subjects_for_this_sem_key:
                    placeholders_sem_subj = ','.join(['%s'] * len(subjects_for_this_sem_key))
                    
                    mat_query = f"SELECT * FROM materials WHERE teacher = %s AND semester = %s AND subject IN ({placeholders_sem_subj}) ORDER BY upload_date DESC"
                    cursor.execute(mat_query, [username, sem_taught_key] + subjects_for_this_sem_key)
                    sem_materials_list = cursor.fetchall()
                    for item in sem_materials_list: item['file_url'] = url_for('serve_upload', filename=item['filename'])
                    current_sem_data['materials'].extend(sem_materials_list)
                    teacher_materials.extend(sem_materials_list)

                    assign_query = f"SELECT * FROM assignments WHERE uploaded_by = %s AND semester = %s AND subject IN ({placeholders_sem_subj}) ORDER BY upload_date DESC"
                    cursor.execute(assign_query, [username, sem_taught_key] + subjects_for_this_sem_key)
                    sem_assignments_list = cursor.fetchall()
                    for item in sem_assignments_list: item['file_url'] = url_for('serve_upload', filename=item['filename'])
                    current_sem_data['assignments'].extend(sem_assignments_list)
                    teacher_assignments.extend(sem_assignments_list)

                    tests_query_teacher = f"SELECT * FROM tests WHERE teacher_username = %s AND semester = %s AND subject IN ({placeholders_sem_subj}) ORDER BY created_at DESC"
                    cursor.execute(tests_query_teacher, [username, sem_taught_key] + subjects_for_this_sem_key)
                    semester_tests_list = cursor.fetchall()
                    
                    for test_item_detail in semester_tests_list:
                        cursor.execute("""
                            SELECT 
                                ta.id AS attempt_id, ta.student_id, ta.score, 
                                ta.num_questions_in_attempt, ta.total_marks_possible,
                                ta.status AS attempt_status, ta.started_at, ta.completed_at,
                                u.username AS student_name, u.roll_number 
                            FROM test_attempts ta 
                            JOIN users u ON ta.student_id = u.id 
                            WHERE ta.test_id = %s AND ta.status = 'submitted'
                            ORDER BY ta.score DESC, ta.started_at ASC
                            """, (test_item_detail['id'],))
                        attempts_list_raw = cursor.fetchall()
                        
                        processed_attempts_for_teacher_view = []
                        for raw_attempt_item in attempts_list_raw:
                            attempt_item = dict(raw_attempt_item) 

                            
                            attempt_item['total_questions'] = attempt_item.get('num_questions_in_attempt', 0)
                            attempt_item['attempted_at'] = attempt_item.get('started_at')

                            percentage = None
                            score = attempt_item.get('score')
                            
                            
                            if score is not None:
                                if attempt_item.get('total_marks_possible') and attempt_item['total_marks_possible'] > 0:
                                    percentage = (score / attempt_item['total_marks_possible']) * 100
                                elif attempt_item['total_questions'] and attempt_item['total_questions'] > 0:
                                    percentage = (score / attempt_item['total_questions']) * 100
                            attempt_item['percentage'] = percentage

                            if percentage is not None:
                                if 0 <= percentage <= 30:
                                    attempt_item['level'] = 'Novice'
                                elif 31 <= percentage <= 60:
                                    attempt_item['level'] = 'Intermediate'
                                elif 61 <= percentage <= 100:
                                    attempt_item['level'] = 'Advance'
                                else:
                                    attempt_item['level'] = 'N/A (Invalid %)'
                            else:
                                attempt_item['level'] = 'N/A'
                            processed_attempts_for_teacher_view.append(attempt_item)
                        
                        current_sem_data['tests_with_results'].append({'details': test_item_detail, 'attempts': processed_attempts_for_teacher_view})

        elif role == 'admin':
            cursor.execute("SELECT id, username, email, status FROM users WHERE role = 'teacher'")
            all_teachers_raw = cursor.fetchall()
            teachers_list = [t for t in all_teachers_raw if t['status'] == 'approved']
            pending_teachers_list = [t for t in all_teachers_raw if t['status'] == 'pending']

        return render_template(
            'dashboard.html',
            username=username, role=role, user=user_details, user_status=status,
            roll=roll, department=student_department, semester=student_semester, subjects=subjects_list_for_student,
            student_level=student_level,
            grouped_assignments=grouped_assignments,
            grouped_materials=grouped_materials,
            tests=pending_tests_for_dashboard,
            active_lectures_for_student=active_lectures_for_student,
            semesters_taught=semesters_taught_list,
            subjects_per_semester=subjects_per_semester_data,
            teacher_semester_details=teacher_semester_details,
            assignments=teacher_assignments, 
            materials=teacher_materials,     
            teacher_active_lectures=teacher_active_lectures,
            teachers_list=teachers_list,
            pending_teachers_list=pending_teachers_list,
            datetime=datetime 
        )

    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        if conn and conn.open:
            if 'cursor' in locals() and cursor: cursor.close() 
            conn.close()
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('login'))
    finally:
        if conn and conn.open:
            if 'cursor' in locals() and cursor: cursor.close() 
            conn.close()
    

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    """Handle user logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect('/login')


@app.route('/complete-profile', methods=['GET', 'POST'])
def complete_profile():
    """Handle student profile completion"""
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Please login as a student first', 'error')
        return redirect('/login')

    user_id = session['user_id']
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        if request.method == 'POST':
            roll = request.form['roll']
            department = request.form['department']
            semester = request.form['semester']
            subjects = request.form.getlist('subjects')
            
            subjects_str = ','.join(subjects)
            
            cursor.execute("""
                UPDATE users 
                SET roll_number=%s, department=%s, semester=%s, subjects=%s 
                WHERE id=%s
            """, (roll, department, semester, subjects_str, user_id))
            
            session['department'] = department
            session['semester'] = semester
            
            connection.commit()
            flash('Profile updated successfully!', 'success')
            return redirect('/dashboard')

      
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user = cursor.fetchone()
        
        all_departments = []
        available_semesters_for_user_dept = []
        available_subjects_for_user_sem = []
        semester_display = user.get('semester', '')

        try:
            excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
            logger.info(f"Attempting to load Excel file from: {excel_path}")
            
            if not os.path.exists(excel_path):
                logger.error(f"Excel file NOT FOUND at: {excel_path}")
                flash("System configuration error: The academic data file is missing.", "error")
            else:
                df = pd.read_excel(excel_path)
                df.columns = df.columns.str.strip() 
                logger.info(f"Excel DataFrame columns after stripping: {df.columns.tolist()}")

                
                if 'Department' in df.columns:
                    logger.info("Found 'Department' column in Excel.")
                    
                    
                    raw_departments_from_excel = df['Department'].tolist()
                    logger.debug(f"Raw 'Department' column data (first 20 entries): {raw_departments_from_excel[:20]}") 

                    
                    all_departments = sorted(df['Department'].astype(str).str.strip().unique().tolist())
                    logger.debug(f"Departments after astype(str).strip().unique(): {all_departments}")
                    
                    
                    all_departments = [dept for dept in all_departments if dept and dept.lower() != 'nan']
                    logger.info(f"Final 'all_departments' list prepared for template: {all_departments}") 
                else:
                    logger.error("Column 'Department' NOT FOUND in Excel. 'all_departments' will be empty.") 
                    flash("Configuration error: Could not load department list from source.", "error")

                user_department = user.get('department')
                user_semester = user.get('semester')

                if user_department and 'Department' in df.columns and 'Semester' in df.columns:
                    sem_df = df[df['Department'].astype(str).str.strip().str.lower() == user_department.lower().strip()]
                    available_semesters_for_user_dept = sorted(sem_df['Semester'].astype(str).str.strip().unique().tolist())
                    available_semesters_for_user_dept = [sem for sem in available_semesters_for_user_dept if sem and sem.lower() != 'nan']

                    if user_semester and 'Subject' in df.columns:
                        subj_df = sem_df[sem_df['Semester'].astype(str).str.strip().str.lower() == user_semester.lower().strip()]
                        available_subjects_for_user_sem = sorted(subj_df['Subject'].astype(str).str.strip().unique().tolist())
                        available_subjects_for_user_sem = [subj for subj in available_subjects_for_user_sem if subj and subj.lower() != 'nan']
            
            if user_semester: 
                semester_display_val = str(user_semester).lower().replace('semester', '').strip()
                if semester_display_val == '1': semester_display = '1st'
                
                else: semester_display = user_semester
            
        except FileNotFoundError: 
            logger.error(f"Excel file not found at: {excel_path} in complete_profile (caught by FileNotFoundError)")
            flash("System configuration error: Subject data file not found.", 'error')
        except Exception as e:
            logger.error(f"Error loading data from Excel in complete_profile: {str(e)}", exc_info=True)
            flash(f"Error loading profile options: {str(e)}", 'error')

        connection.close()
        
        user_subjects_str = user.get('subjects')
        if user_subjects_str:
            current_subjects = [s.strip() for s in user_subjects_str.split(',') if s.strip()]
        else:
            current_subjects = []

        return render_template(
            'complete_profile.html',
            user=user,
            all_departments=all_departments,
            available_semesters=available_semesters_for_user_dept,
            available_subjects=available_subjects_for_user_sem,
            semester_display=semester_display,
            current_department=user.get('department'),
            current_semester=user.get('semester'),
            current_subjects=current_subjects
        )
    
    except Exception as e:
        logger.error(f"Complete profile error: {str(e)}", exc_info=True)
        if 'connection' in locals() and connection and connection.open:
            connection.close()
        flash(f'Error loading profile page: {str(e)}', 'error')
        return redirect('/dashboard')



@app.route('/class/<semester>')
def class_management(semester):
    """Class management page for teachers"""
    if 'username' not in session or session['role'] != 'teacher':
        flash('Please login as a teacher', 'error')
        return redirect('/login')
    
    try:
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        cursor.execute("""
            SELECT id, username, roll_number, subjects FROM users 
            WHERE role='student' AND semester=%s
        """, (semester,))
        students = cursor.fetchall()
        
        cursor.execute("""
            SELECT * FROM materials 
            WHERE semester=%s AND teacher=%s
            ORDER BY upload_date DESC
        """, (semester, session['username']))
        materials = cursor.fetchall()
        
        cursor.execute("""
            SELECT * FROM assignments 
            WHERE semester=%s AND uploaded_by=%s
            ORDER BY due_date DESC
        """, (semester, session['username']))
        assignments = cursor.fetchall()
        
        for material in materials:
            material['file_url'] = f"/uploads/{material['filename']}"
        
        for assignment in assignments:
            assignment['file_url'] = f"/uploads/{assignment['filename']}"
        
        connection.close()
        
        return render_template('class_management.html',
                             semester=semester,
                             students=students,
                             materials=materials,
                             assignments=assignments)
    
    except Exception as e:
        logger.error(f"Class management error: {str(e)}")
        flash(f'Error loading class data: {str(e)}', 'error')
        return redirect('/dashboard')


@app.route('/upload-content', methods=['POST'])
def upload_content():
    if 'username' not in session or session['role'] != 'teacher':
        flash('Please login as a teacher', 'error')
        return redirect('/login')

    conn = None
    try:
   
        semester = request.form.get('semester')
        subject = request.form.get('subject')
        content_type = request.form.get('content_type')
        title = request.form.get('title')
        description = request.form.get('description', '')
        due_date = request.form.get('due_date') or None
        file = request.files.get('file')
        
        
        difficulty_level = request.form.get('difficulty_level')

        logger.info(f"Upload Content: subject={subject}, semester={semester}, content_type={content_type}, title={title}, difficulty={difficulty_level}")

       
        if not all([semester, subject, content_type, title, difficulty_level, file and file.filename]):
            flash('All fields including Difficulty Level and a file are required', 'error')
            return redirect('/dashboard')

        
        allowed_difficulty_levels = ['Novice', 'Intermediate', 'Advance', 'General']
        if difficulty_level not in allowed_difficulty_levels:
            flash(f'Invalid difficulty level selected. Allowed levels are: {", ".join(allowed_difficulty_levels)}.', 'error')
            return redirect('/dashboard') 

        department = session.get('department')
        if not department: 
            user_details_conn = get_db_connection()
            user_cursor = user_details_conn.cursor()
            user_cursor.execute("SELECT department FROM users WHERE id = %s", (session['user_id'],))
            user_dept_data = user_cursor.fetchone()
            user_cursor.close()
            user_details_conn.close()
            if user_dept_data and user_dept_data.get('department'):
                department = user_dept_data.get('department').split(',')[0].strip() 
            else:
                flash('Your department information is missing. Please complete your profile.', 'error')
                return redirect('/dashboard')
        elif isinstance(department, str) and ',' in department: 
             department = department.split(',')[0].strip()


        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], department, subject, content_type) 
        os.makedirs(upload_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        filename = f"{timestamp}_{original_filename}"
        filepath = os.path.join(upload_path, filename)
        file.save(filepath)

        stored_filename = os.path.join(department, subject, content_type, filename).replace('\\', '/') 
        logger.info(f"File saved at {filepath}. Storing in DB as: {stored_filename}")

        conn = get_db_connection()
        cursor = conn.cursor()

        if content_type == 'assignment':
          
            cursor.execute("""
                INSERT INTO assignments 
                (semester, subject, title, description, filename, uploaded_by, upload_date, due_date, difficulty_level)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            """, (semester, subject, title, description, stored_filename, session['username'], due_date, difficulty_level))
            logger.info(f"Inserted assignment: {title}, Difficulty: {difficulty_level}")
        else: 
            cursor.execute("""
                INSERT INTO materials 
                (semester, subject, title, description, filename, teacher, upload_date, type, difficulty_level)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
            """, (semester, subject, title, description, stored_filename, session['username'], content_type, difficulty_level))
            logger.info(f"Inserted material: {title} (Type: {content_type}), Difficulty: {difficulty_level}")

        conn.commit()
        flash('Content uploaded successfully!', 'success')

    except pymysql.Error as db_err:
        logger.error(f"Database error during content upload: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error during upload: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if 'conn' in locals() and conn and conn.open: 
            conn.rollback()
        flash(f'Upload failed: {str(e)}', 'error')
    finally:
        if 'conn' in locals() and conn and conn.open:
            cursor.close()
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/subject/<subject_name>')
def view_subject(subject_name):
    if 'username' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    role = session.get('role')
    current_semester_from_session = session.get('semester') 
    
    subject_input_normalized = subject_name.replace('-', ' ').strip().lower()
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        student_level = None
        if role == 'student' and user_id:
            cursor.execute("SELECT level FROM users WHERE id = %s", (user_id,))
            student_user_details = cursor.fetchone()
            if student_user_details:
                student_level = student_user_details.get('level')
        logger.info(f"Viewing subject '{subject_input_normalized}', Student Level: {student_level}")

        cursor.execute("""
            SELECT DISTINCT subject FROM (
                SELECT subject FROM materials WHERE semester = %s
                UNION
                SELECT subject FROM assignments WHERE semester = %s
                UNION
                SELECT subject FROM tests WHERE semester = %s
            ) AS all_content_subjects
            WHERE LOWER(REPLACE(REPLACE(subject, '-', ' '), '&', 'and')) = %s OR LOWER(subject) = %s
            LIMIT 1
        """, (current_semester_from_session, current_semester_from_session, current_semester_from_session, 
              subject_input_normalized.replace('&', 'and'), subject_input_normalized))
        subject_match = cursor.fetchone()
        
        
        real_subject_name_for_query = subject_match['subject'] if subject_match else subject_input_normalized.title()
        logger.info(f"Resolved subject name for queries: '{real_subject_name_for_query}' (normalized from URL: '{subject_input_normalized}')")

        
        difficulty_filter_sql_part = ""
        difficulty_params_list = []
        if role == 'student':
            if student_level:
                difficulty_filter_sql_part = " AND (difficulty_level = %s OR difficulty_level = 'General' OR difficulty_level IS NULL OR difficulty_level = '')"
                difficulty_params_list.append(student_level)
            else: 
                difficulty_filter_sql_part = " AND (difficulty_level = 'Novice' OR difficulty_level = 'General' OR difficulty_level IS NULL OR difficulty_level = '')"

       
        assignments_sql = """
            SELECT a.*, u.username as teacher_name 
            FROM assignments a
            JOIN users u ON a.uploaded_by = u.username
            WHERE LOWER(a.subject) = %s AND a.semester = %s
        """
        assignments_params = [real_subject_name_for_query.lower(), current_semester_from_session]
        if role == 'student':
            assignments_sql += difficulty_filter_sql_part.replace('difficulty_level', 'a.difficulty_level') # Alias for assignments
            assignments_params.extend(difficulty_params_list)
        assignments_sql += " ORDER BY a.upload_date DESC"
        cursor.execute(assignments_sql, assignments_params)
        assignments = cursor.fetchall()
        for a_item in assignments: 
            a_item['file_url'] = url_for('serve_upload', filename=a_item['filename']) if a_item.get('filename') else None

        
        materials_sql = """
            SELECT m.*, u.username as teacher_name_from_users_table 
            FROM materials m
            JOIN users u ON m.teacher = u.username
            WHERE LOWER(m.subject) = %s AND m.semester = %s
        """
        materials_params = [real_subject_name_for_query.lower(), current_semester_from_session]
        if role == 'student':
            materials_sql += difficulty_filter_sql_part.replace('difficulty_level', 'm.difficulty_level') 
            materials_params.extend(difficulty_params_list)
        materials_sql += " ORDER BY m.upload_date DESC"
        cursor.execute(materials_sql, materials_params)
        materials = cursor.fetchall()
        for m_item in materials: 
            m_item['file_url'] = url_for('serve_upload', filename=m_item['filename']) if m_item.get('filename') else None
            m_item['teacher_display_name'] = m_item.get('teacher_name_from_users_table', m_item.get('teacher'))


        subject_tests_with_attempts = []
        if role == 'student' and user_id:
            tests_sql = """
                SELECT t.*, t.teacher_username as teacher_name
                FROM tests t
                WHERE LOWER(t.subject) = %s AND t.semester = %s
            """
            tests_params = [real_subject_name_for_query.lower(), current_semester_from_session]
           
            tests_sql += difficulty_filter_sql_part.replace('difficulty_level', 't.difficulty_level') 
            tests_params.extend(difficulty_params_list)
            
            tests_sql += " ORDER BY t.due_date ASC, t.created_at DESC"
            cursor.execute(tests_sql, tests_params)
            subject_tests_raw = cursor.fetchall()

            if subject_tests_raw:
                for test_item_dict_raw in subject_tests_raw:
                    test_item_dict = dict(test_item_dict_raw) 
                    cursor.execute("""
                        SELECT id, score, total_marks_possible, num_questions_in_attempt, started_at 
                        FROM test_attempts
                        WHERE student_id = %s AND test_id = %s
                        ORDER BY started_at DESC LIMIT 1
                    """, (user_id, test_item_dict['id']))
                    attempt_info = cursor.fetchone()
                    
                    test_item_dict['attempted'] = bool(attempt_info)
                    if attempt_info:
                        test_item_dict['attempt_id'] = attempt_info['id']
                        test_item_dict['student_score'] = attempt_info['score']
                        
                        test_item_dict['total_marks_in_attempt'] = attempt_info.get('total_marks_possible') or test_item_dict.get('total_marks_possible') 
                        test_item_dict['num_questions_in_attempt'] = attempt_info.get('num_questions_in_attempt') or test_item_dict.get('total_questions') 
                        test_item_dict['attempt_started_at'] = attempt_info['started_at']
                    else: 
                        
                        test_item_dict['total_marks_in_attempt'] = test_item_dict.get('total_marks_possible') 
                        test_item_dict['num_questions_in_attempt'] = test_item_dict.get('total_questions') 
                    subject_tests_with_attempts.append(test_item_dict)
        
        return render_template(
            "subject_view.html",
            subject=real_subject_name_for_query,
            assignments=assignments,
            materials=materials,
            subject_tests=subject_tests_with_attempts, 
            datetime=datetime, 
            role=role,
            student_level=student_level
        )

    except Exception as e:
        logger.error(f"Subject view error for '{subject_name}': {str(e)}", exc_info=True)
        flash(f"Error loading subject '{subject_name}': {str(e)}", 'error')
        return redirect(url_for('dashboard'))
    finally:
        if conn: 
            if 'cursor' in locals() and cursor: cursor.close()
            conn.close()
    

@app.route('/complete-teacher-profile', methods=['GET', 'POST'])
def complete_teacher_profile():
    """Handle teacher profile completion"""
    if 'user_id' not in session or session['role'] != 'teacher':
        flash('Please login as a teacher first', 'error')
        return redirect('/login')

    user_id = session['user_id']
    connection = None
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        if request.method == 'POST':
            try:
                departments = request.form.getlist('departments')
                semesters = request.form.getlist('semesters')
                subjects = request.form.getlist('teaching_subjects')
                
                
                semester_subjects = {}
                for sem in semesters:
                    
                    sem_subjects = [sub for sub in subjects if sub in get_subjects_for_semester(sem, departments)]
                    if sem_subjects:  
                        semester_subjects[sem] = sem_subjects
                
                
                subjects_str = ""
                for sem, subjs in semester_subjects.items():
                    subjects_str += f"{sem}: {', '.join(subjs)}; "
                
               
                subjects_str = subjects_str.rstrip('; ')
                
                departments_str = ','.join(departments)
                semesters_str = ','.join(semesters)
                
                
                cursor.execute("""
                    UPDATE users 
                    SET department=%s, 
                        departments_taught=%s,
                        semesters_taught=%s, 
                        subjects_per_semester=%s
                    WHERE id=%s
                """, (departments_str, departments_str, semesters_str, subjects_str, user_id))
                
                
                session['department'] = departments_str
                
                connection.commit()
                flash('Profile updated successfully!', 'success')
                return redirect('/dashboard')
                
            except Exception as e:
                if connection:
                    connection.rollback()
                logger.error(f"Error saving profile: {str(e)}")
                flash(f"Error saving profile: {str(e)}", 'error')
                return redirect(request.url)
        
        else:  
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            
            excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
            df = pd.read_excel(excel_path)
            available_departments = sorted(df['Department'].str.strip().unique().tolist())
            available_semesters = sorted(df['Semester'].astype(str).unique().tolist())
            
            return render_template(
                'complete_teacher_profile.html',
                user=user,
                available_departments=available_departments,
                available_semesters=available_semesters
            )
        
    except Exception as e:
        logger.error(f"Error loading teacher profile data: {str(e)}")
        flash(f"Error loading profile data: {str(e)}", 'error')
        return redirect('/dashboard')
        
    finally:
        if connection:
            connection.close()



@app.route('/get-semesters-for-department', methods=['GET'])
def get_semesters_for_department():
    department = request.args.get('department')
    if not department:
        return jsonify({'error': 'Department is required'}), 400

    semesters = []
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
        df = pd.read_excel(excel_path)
        df.columns = df.columns.str.strip() 

        if 'Department' in df.columns and 'Semester' in df.columns:
           
            filtered_df = df[df['Department'].astype(str).str.strip().str.lower() == department.lower().strip()]
            
            semesters = sorted(filtered_df['Semester'].astype(str).str.strip().unique().tolist())
            semesters = [sem for sem in semesters if sem] 
            logger.error("Excel file is missing 'Department' or 'Semester' column for /get-semesters-for-department.")
            return jsonify({'error': 'Server configuration error: Could not retrieve semester list.'}), 500
            
        return jsonify({'semesters': semesters})
    except FileNotFoundError:
        logger.error(f"Excel file not found at: {excel_path} in /get-semesters-for-department")
        return jsonify({'error': 'Server configuration error: Data file not found.'}), 500
    except Exception as e:
        logger.error(f"Error in /get-semesters-for-department: {str(e)}")
        return jsonify({'error': str(e)}), 500



def get_subjects_for_semester(semester, departments):
    """Get subjects for a specific semester and departments from Excel"""
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
        df = pd.read_excel(excel_path)
        
        filtered = df[
            (df['Semester'].astype(str).str.strip() == str(semester).strip()) &
            (df['Department'].str.strip().str.lower().isin(
                [d.lower().strip() for d in departments]
            ))
        ]
        return filtered['Subject'].unique().tolist()
    except Exception as e:
        logger.error(f"Error getting subjects: {str(e)}")
        return []

@app.route('/get-all-subjects')
def get_all_subjects():
    """Get all subjects for given semesters and departments"""
    semesters = request.args.get('semesters', '').split(',')
    departments = request.args.get('departments', '').split(',')
    
    
    semesters = [s.strip() for s in semesters if s.strip()]
    departments = [d.strip() for d in departments if d.strip()]
    
    if not semesters or not departments:
        return {'error': 'No semesters or departments specified'}, 400
    
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')
        df = pd.read_excel(excel_path)
        
        
        df['Department'] = df['Department'].str.strip().str.upper()
        departments = [d.strip().upper() for d in departments]
        
        
        filtered = df[
            (df['Semester'].astype(str).isin(semesters)) &
            (df['Department'].isin(departments))
        ]
       
        subjects = filtered['Subject'].unique().tolist()
        
        return {'subjects': subjects}
    except Exception as e:
        logger.error(f"Error in get-all-subjects: {str(e)}")
        return {'error': str(e)}, 500


@app.route('/create-test', methods=['POST'])
def create_test():
    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher to create tests.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        teacher_id = session['user_id']
        teacher_username = session['username']
        title = request.form.get('title')
        subject = request.form.get('subject')
        semester = request.form.get('semester')
        due_date_str = request.form.get('due_date')
        test_type_overall = request.form.get('test_type')
        
        difficulty_level = request.form.get('difficulty_level')
        question_count = int(request.form.get('question_count', 0))

        logger.info(f"Create Test: Title={title}, Subject={subject}, Sem={semester}, DueDate={due_date_str}, TestTypeOverall={test_type_overall}, Difficulty={difficulty_level}, QCount={question_count}")

        if not all([title, subject, semester, due_date_str, test_type_overall, difficulty_level]):
            flash('Please fill in all test details (Title, Subject, Semester, Due Date, Test Type, Difficulty Level).', 'error')
            return redirect(url_for('dashboard'))

        allowed_difficulty_levels = ['Novice', 'Intermediate', 'Advance', 'General']
        if difficulty_level not in allowed_difficulty_levels:
            flash(f'Invalid difficulty level selected. Allowed levels are: {", ".join(allowed_difficulty_levels)}.', 'error')
            return redirect(url_for('dashboard'))

        if question_count <= 0:
            flash('A test must have at least one question.', 'error')
            return redirect(url_for('dashboard'))

        try:
            due_date = datetime.strptime(due_date_str, '%Y-%m-%dT%H:%M') if due_date_str else None
        except ValueError:
            flash('Invalid due date format.', 'error')
            return redirect(url_for('dashboard'))

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tests (teacher_id, teacher_username, title, subject, semester, due_date, test_type, difficulty_level, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (teacher_id, teacher_username, title, subject, semester, due_date, test_type_overall, difficulty_level))

        test_id = cursor.lastrowid
        logger.info(f"Test created with ID: {test_id}, Difficulty: {difficulty_level}")

        question_images_base_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'question_images')
        test_image_upload_path = os.path.join(question_images_base_dir, f'test_{test_id}')
        os.makedirs(test_image_upload_path, exist_ok=True)

        for i in range(1, question_count + 1):
            question_type_specific = request.form.get(f'q{i}_type')
            topic = (request.form.get(f'q{i}_topic') or '').strip()
            image_file = request.files.get(f'q{i}_image')
            image_db_path = None

            if image_file and image_file.filename:
                allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif'}
                filename, file_extension = os.path.splitext(image_file.filename)
                if file_extension.lower() not in allowed_extensions:
                    flash(f'Error in question {i}: Invalid image file type ({file_extension}). Allowed: PNG, JPG, JPEG, GIF.', 'error')
                    if conn: conn.rollback()
                    return redirect(url_for('dashboard'))
                
                original_img_filename = secure_filename(image_file.filename)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                image_disk_filename = f"q{i}_{timestamp}_{original_img_filename}"
                full_image_path_on_disk = os.path.join(test_image_upload_path, image_disk_filename)
                try:
                    image_file.save(full_image_path_on_disk)
                    
                    image_db_path = os.path.join('question_images', f'test_{test_id}', image_disk_filename).replace('\\\\', '/')
                    logger.info(f"SUCCESS: Saved image for Q{i} of Test ID {test_id} to disk at {full_image_path_on_disk}, DB path: {image_db_path}")
                except Exception as save_err:
                    logger.error(f"ERROR: Failed to save image for Q{i} of Test ID {test_id}: {save_err}", exc_info=True)
                    flash(f'Error saving image for question {i}.', 'error')
                    if conn: conn.rollback()
                    return redirect(url_for('dashboard'))
            
            q_text, q_a, q_b, q_c, q_d, q_correct = None, None, None, None, None, None
            if question_type_specific == 'mcq':
                q_text = (request.form.get(f'q{i}_text_mcq') or '').strip()
                q_a = (request.form.get(f'q{i}_a') or '').strip()
                q_b = (request.form.get(f'q{i}_b') or '').strip()
                q_c = (request.form.get(f'q{i}_c') or '').strip()
                q_d = (request.form.get(f'q{i}_d') or '').strip()
                q_correct = (request.form.get(f'q{i}_correct_mcq') or '').strip().upper()
                if not all([q_text, q_a, q_b, q_c, q_d, q_correct]) or q_correct not in ['A', 'B', 'C', 'D']:
                    flash(f'Error in MCQ question {i}. All fields must be filled and correct option valid.', 'error')
                    if conn: conn.rollback()
                    return redirect(url_for('dashboard'))
            elif question_type_specific == 'fill_blank':
                q_text = (request.form.get(f'q{i}_text_fill') or '').strip()
                q_correct = (request.form.get(f'q{i}_answer_fill') or '').strip()
                if not all([q_text, q_correct]):
                    flash(f'Error in Fill in the Blanks question {i}. Question and answer must be filled.', 'error')
                    if conn: conn.rollback()
                    return redirect(url_for('dashboard'))
            elif question_type_specific == 'true_false':
                q_text = (request.form.get(f'q{i}_text_tf') or '').strip()
                q_correct = request.form.get(f'q{i}_correct_tf')
                if not q_text or q_correct not in ['True', 'False']:
                    flash(f'Error in True/False question {i}. Statement and answer must be provided and valid.', 'error')
                    if conn: conn.rollback()
                    return redirect(url_for('dashboard'))
            else:
                flash(f'Invalid question type for question {i}.', 'error')
                if conn: conn.rollback()
                return redirect(url_for('dashboard'))

            cursor.execute("""
                INSERT INTO questions (test_id, question_text, option_a, option_b, option_c, option_d, correct_option, topic, question_type, image_filename)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (test_id, q_text, q_a, q_b, q_c, q_d, q_correct, topic, question_type_specific, image_db_path))
        
        conn.commit()
        flash('Test created successfully!', 'success')
        return redirect(url_for('dashboard'))

    except pymysql.Error as db_err:
        logger.error(f"Database error in create_test: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error: {db_err}', 'error')
    except ValueError as val_err: 
        logger.error(f"Value error in create_test: {val_err}", exc_info=True)
        flash(f'Invalid data format: {val_err}', 'error')
    except Exception as e:
        logger.error(f"Unexpected error in create_test: {str(e)}", exc_info=True)
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn:
            conn.close()
    return redirect(url_for('dashboard'))


@app.route('/assignment/<int:assignment_id>/submissions')
def view_assignment_submissions(assignment_id):
    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher to view submissions.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        
        cursor.execute("SELECT * FROM assignments WHERE id = %s AND uploaded_by = %s", (assignment_id, session['username']))
        assignment = cursor.fetchone()

        if not assignment:
            flash('Assignment not found or you are not authorized to view its submissions.', 'error')
            return redirect(url_for('dashboard'))

       
        cursor.execute("""
            SELECT s.*, u.username AS student_name, u.roll_number AS student_roll_number
            FROM submissions s 
            JOIN users u ON s.student_id = u.id 
            WHERE s.assignment_id = %s
            ORDER BY s.submission_date DESC
        """, (assignment_id,))
        submissions = cursor.fetchall()

        
        for sub in submissions:
            if sub.get('submission_file'):
                sub['file_url'] = url_for('serve_upload', filename=sub['submission_file'])

        return render_template('view_submissions.html', assignment=assignment, submissions=submissions)

    except pymysql.Error as db_err:
        logger.error(f"Database error viewing submissions: {db_err}")
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error viewing submissions: {str(e)}")
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn:
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/submit-assignment/<int:assignment_id>', methods=['GET', 'POST'])
def submit_assignment(assignment_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Please login as a student to submit assignments.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM assignments WHERE id = %s", (assignment_id,))
        assignment = cursor.fetchone()

        if not assignment:
            flash('Assignment not found.', 'error')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            student_id = session['user_id']
            submitted_file = request.files.get('submission_file')

            if not submitted_file or not submitted_file.filename:
                flash('Please select a file to submit.', 'error')
                return redirect(url_for('submit_assignment', assignment_id=assignment_id))

            
            student_dept = session.get('department', 'general_students') 
            student_semester = session.get('semester', 'unknown_semester') 

            submission_subfolder = os.path.join(
                'submissions', 
                f"dept_{student_dept}",
                f"sem_{student_semester}",
                f"student_{student_id}",
                f"assignment_{assignment_id}"
            )
            student_submission_path_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], submission_subfolder)
            os.makedirs(student_submission_path_on_disk, exist_ok=True)
            
            original_filename = secure_filename(submitted_file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_to_store_on_disk = f"{timestamp}_{original_filename}" 
            
            filepath_on_disk = os.path.join(student_submission_path_on_disk, filename_to_store_on_disk)
            submitted_file.save(filepath_on_disk)

            
            db_relative_filepath = os.path.join(submission_subfolder, filename_to_store_on_disk)


            cursor.execute("SELECT id FROM submissions WHERE assignment_id = %s AND student_id = %s", (assignment_id, student_id))
            existing_submission = cursor.fetchone()

            if existing_submission:
                cursor.execute("""
                    UPDATE submissions 
                    SET submission_file = %s, filename = %s, submission_date = NOW()
                    WHERE id = %s
                """, (db_relative_filepath, original_filename, existing_submission['id'])) 
                flash('Assignment re-submitted successfully!', 'success')
            else:
                cursor.execute("""
                    INSERT INTO submissions (assignment_id, student_id, submission_file, filename, submission_date)
                    VALUES (%s, %s, %s, %s, NOW())
                """, (assignment_id, student_id, db_relative_filepath, original_filename)) 
            
            conn.commit()
            return redirect(url_for('dashboard'))

        return render_template('submit_assignment.html', assignment=assignment)

    except pymysql.Error as db_err:
        logger.error(f"Database error during assignment submission: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error submitting assignment: {str(e)}", exc_info=True)
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn: conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/test-results/<int:test_id>')
def view_test_results(test_id):
    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher to view test results.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM tests 
            WHERE id = %s AND teacher_username = %s
        """, (test_id, session['username']))
        test_details = cursor.fetchone()

        if not test_details:
            flash('Test not found or you are not authorized to view its results.', 'error')
            return redirect(url_for('dashboard'))

        
        cursor.execute("""
            SELECT 
                ta.id AS attempt_id, 
                ta.student_id, 
                ta.score, 
                ta.num_questions_in_attempt,  -- Crucial for 'Total Questions'
                ta.total_marks_possible,      -- Useful for percentage if marks vary
                ta.status AS attempt_status, 
                ta.started_at,                -- For 'Attempted At'
                ta.completed_at,
                u.username AS student_name 
            FROM test_attempts ta
            JOIN users u ON ta.student_id = u.id
            WHERE ta.test_id = %s AND ta.status = 'submitted' 
            ORDER BY ta.score DESC, ta.started_at ASC
        """, (test_id,))
        
        raw_attempts = cursor.fetchall()
        processed_attempts = []

        for raw_attempt in raw_attempts:
            attempt = dict(raw_attempt) 

       
            attempt['total_questions'] = attempt.get('num_questions_in_attempt', 0)

            attempt['attempted_at'] = attempt.get('started_at') 

            
            percentage = None
            score = attempt.get('score')
        
            denominator_for_percentage = 0
            if attempt.get('total_marks_possible') and attempt['total_marks_possible'] > 0:
                denominator_for_percentage = attempt['total_marks_possible']
            elif attempt['total_questions'] and attempt['total_questions'] > 0: 
                denominator_for_percentage = attempt['total_questions']

            if score is not None and denominator_for_percentage > 0:
                percentage = (float(score) / float(denominator_for_percentage)) * 100
                attempt['percentage'] = percentage
            else:
                attempt['percentage'] = None 

           
            if percentage is not None:
                if 0 <= percentage <= 30:
                    attempt['level'] = 'Novice'
                elif 31 <= percentage <= 60:
                    attempt['level'] = 'Intermediate'
                elif 61 <= percentage <= 100:
                    attempt['level'] = 'Advance'
                else:
                    
                    attempt['level'] = 'N/A (Invalid %)' 
            else:
                attempt['level'] = 'N/A' 
            
            processed_attempts.append(attempt)

        return render_template('view_test_results.html', test=test_details, attempts=processed_attempts)

    except pymysql.Error as db_err:
        logger.error(f"Database error viewing test results for test_id {test_id}: {db_err}", exc_info=True)
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error viewing test results for test_id {test_id}: {str(e)}", exc_info=True)
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn:
            if 'cursor' in locals() and cursor:
                cursor.close()
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/class/<string:semester>/subject/<string:subject_name_url>')
def class_subject_view(semester, subject_name_url):
    logger.info(f"--- Entering class_subject_view ---")
    logger.info(f"Received semester: {semester}, subject_name_url: {subject_name_url}")

    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher first', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

         
        cursor.execute("SELECT semesters_taught, subjects_per_semester FROM users WHERE id = %s", (session['user_id'],))
        teacher_profile = cursor.fetchone()
        if not teacher_profile:
            flash("Teacher profile not found.", "error")
            return redirect(url_for('dashboard'))

        raw_semesters_taught = teacher_profile.get('semesters_taught', '')
        raw_sps = teacher_profile.get('subjects_per_semester', '')

        
        parsed_semesters_taught_list = []
        if raw_semesters_taught:
            parsed_semesters_taught_list = sorted(list(set(s.strip() for s in raw_semesters_taught.split(',') if s.strip())))

        teacher_subjects_per_semester_data = {}
        if raw_sps:
            try:
                if raw_sps.startswith('{') and raw_sps.endswith('}'):
                    processed_sps_json_str = raw_sps.replace("'", '"')
                    try:
                        parsed_json = json.loads(processed_sps_json_str)
                        for sem_key_json, subs_json in parsed_json.items():
                            sem_key_clean = sem_key_json.strip()
                            if isinstance(subs_json, str):
                                teacher_subjects_per_semester_data[sem_key_clean] = [s.strip() for s in subs_json.split(',') if s.strip()]
                            elif isinstance(subs_json, list):
                                teacher_subjects_per_semester_data[sem_key_clean] = [str(s).strip() for s in subs_json if str(s).strip()]
                    except json.JSONDecodeError:
                        teacher_subjects_per_semester_data = {} 

                if not teacher_subjects_per_semester_data and ';' in raw_sps:
                    for block in raw_sps.split(';'):
                        block = block.strip()
                        if ':' in block:
                            sem, subs_str = block.split(':', 1)
                            sem_key = sem.strip()
                            if sem_key: teacher_subjects_per_semester_data[sem_key] = [s.strip() for s in subs_str.split(',') if s.strip()]
                
                elif not teacher_subjects_per_semester_data and ':' in raw_sps:
                    sem, subs_str = raw_sps.split(':', 1)
                    sem_key = sem.strip()
                    if sem_key: teacher_subjects_per_semester_data[sem_key] = [s.strip() for s in subs_str.split(',') if s.strip()]
                
                elif not teacher_subjects_per_semester_data and parsed_semesters_taught_list and len(parsed_semesters_taught_list) == 1:
                    teacher_subjects_per_semester_data[parsed_semesters_taught_list[0]] = [s.strip() for s in raw_sps.split(',') if s.strip()]
                
                elif not teacher_subjects_per_semester_data and raw_sps:
                     logger.warning(f"CSubjView: Could not parse 'subjects_per_semester' ('{raw_sps}') for teacher {session['username']}.")
            except Exception as e_parse_sps_view:
                logger.error(f"CSubjView: Exception parsing subjects_per_semester: {e_parse_sps_view}", exc_info=True)
                teacher_subjects_per_semester_data = {}
        
        logger.debug(f"CSubjView: Parsed teacher subjects for this view: {teacher_subjects_per_semester_data}")


        canonical_teacher_subject_name = None
        teacher_subjects_for_this_semester = teacher_subjects_per_semester_data.get(semester, [])
        
        for original_subj_name in teacher_subjects_for_this_semester:
    
            temp_slug = original_subj_name.replace('&', 'and').replace(' ', '-').lower()
            if temp_slug == subject_name_url:
                canonical_teacher_subject_name = original_subj_name 
                break
        
        if not canonical_teacher_subject_name:
            logger.error(f"CSubjView: Could not find a matching canonical subject for slug '{subject_name_url}' in semester '{semester}' for teacher {session['username']}.")
            flash("Error: Subject not found or mapping issue.", "error")
            return redirect(url_for('dashboard'))

        
        normalized_subject_for_comparison = canonical_teacher_subject_name.lower()
        logger.info(f"CSubjView: Canonical subject: '{canonical_teacher_subject_name}', Normalized for comparison: '{normalized_subject_for_comparison}'")

        
        logger.info(f"CSubjView: Fetching all students for semester: {semester}")
        cursor.execute("SELECT id, username, roll_number, email, subjects FROM users WHERE role='student' AND semester=%s", (semester,))
        all_students_in_semester = cursor.fetchall()
        logger.info(f"CSubjView: Found {len(all_students_in_semester) if all_students_in_semester else 0} students in semester {semester}.")

        enrolled_students = []
        if all_students_in_semester:
            for student in all_students_in_semester:
                student_subjects_str = student.get('subjects', '')
                logger.debug(f"CSubjView STUDENT_CHECK: ID {student.get('id')} ({student.get('username')}), DB_Subj: '{student_subjects_str}'")
                if student_subjects_str:
                    student_subjects_list = [s.strip().lower() for s in student_subjects_str.split(',') if s.strip()]
                    logger.debug(f"CSubjView STUDENT_CHECK: ID {student.get('id')} Norm_Subj_List: {student_subjects_list}")
                    logger.debug(f"CSubjView STUDENT_CHECK: Comparing Teacher's '{normalized_subject_for_comparison}' with Student's list.")
                    if normalized_subject_for_comparison in student_subjects_list:
                        enrolled_students.append(student)
                        logger.info(f"CSubjView STUDENT_MATCH: ID {student.get('id')} MATCHED '{normalized_subject_for_comparison}'.")
            
        
        logger.info(f"CSubjView: Number of enrolled_students found for '{canonical_teacher_subject_name}': {len(enrolled_students)}")


        logger.info(f"CSubjView: Fetching materials for sem='{semester}', subj_norm='{normalized_subject_for_comparison}', teacher='{session['username']}'")
        cursor.execute("""
            SELECT * FROM materials
            WHERE semester=%s AND LOWER(subject)=%s AND teacher=%s 
            ORDER BY upload_date DESC
        """, (semester, normalized_subject_for_comparison, session['username']))
        materials = cursor.fetchall()
        for material_item in materials: material_item['file_url'] = url_for('serve_upload', filename=material_item['filename'])
        logger.info(f"CSubjView: Found {len(materials)} materials.")

        logger.info(f"CSubjView: Fetching assignments for sem='{semester}', subj_norm='{normalized_subject_for_comparison}', teacher='{session['username']}'")
        cursor.execute("""
            SELECT * FROM assignments
            WHERE semester=%s AND LOWER(subject)=%s AND uploaded_by=%s
            ORDER BY due_date DESC
        """, (semester, normalized_subject_for_comparison, session['username']))
        assignments = cursor.fetchall()
        for assignment_item in assignments: 
            assignment_item['file_url'] = url_for('serve_upload', filename=assignment_item['filename'])
            assignment_item['view_submissions_url'] = url_for('view_assignment_submissions', assignment_id=assignment_item['id'])
        logger.info(f"CSubjView: Found {len(assignments)} assignments.")

        return render_template(
            'teacher_class_subject.html',
            semester=semester,
            subject_name=canonical_teacher_subject_name, 
            enrolled_students=enrolled_students,
            materials=materials,
            assignments=assignments,
            username=session.get('username')
        )

    except pymysql.Error as db_err:
        logger.error(f"CSubjView: Database error for {semester}/{subject_name_url}: {db_err}", exc_info=True)
        flash(f'Database error loading subject data: {db_err}', 'error')
    except Exception as e:
        logger.error(f"CSubjView: Teacher subject view error for {semester}/{subject_name_url}: {str(e)}", exc_info=True)
        flash(f'Error loading subject data: {str(e)}', 'error')
    finally:
        if conn:
            conn.close()
        logger.info(f"--- Exiting class_subject_view (finally or after return) ---")
    
    return redirect(url_for('dashboard'))


@app.route('/take-test/<int:test_id>', methods=['GET', 'POST'])
def take_test_page(test_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Please login as a student to take tests.', 'error')
        return redirect(url_for('login'))

    student_id = session['user_id']
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM test_attempts WHERE student_id = %s AND test_id = %s", (student_id, test_id))
        existing_attempt = cursor.fetchone()
        if existing_attempt:
            flash("You have already attempted this test. View your results.", 'info')
            return redirect(url_for('view_test_attempt_results', attempt_id=existing_attempt['id']))

        
        cursor.execute("SELECT * FROM tests WHERE id = %s", (test_id,))
        test_details = cursor.fetchone()
        if not test_details:
            flash('Test not found.', 'error')
            return redirect(url_for('dashboard'))

       
        cursor.execute(
            "SELECT id, question_text, option_a, option_b, option_c, option_d, "
            "correct_option, topic, question_type, image_filename, marks "
            "FROM questions WHERE test_id = %s ORDER BY id ASC", (test_id,)
        )
        questions_raw = cursor.fetchall() 
       
        questions = []
        if questions_raw:
            for q_raw_item in questions_raw:
                q_item = dict(q_raw_item) 
                if q_item.get('image_filename'):
                   
                    q_item['image_filename'] = q_item['image_filename'].replace('\\', '/')
                    
                questions.append(q_item)
        
        if not questions:
            flash('No questions found for this test. Please contact your teacher.', 'error')
            return redirect(url_for('dashboard'))
        
        actual_question_count = len(questions)

       
        logger.info(f"--- Questions data for test_id {test_id} being passed to template (after normalization): ---")
        for q_idx, q_data in enumerate(questions):
            logger.info(f"Question {q_idx + 1}: {q_data}")
        logger.info("--- End of questions data ---")

        if request.method == 'POST':
            current_score = 0
            current_total_marks_possible = 0
            for q_item_loop in questions: 
                current_total_marks_possible += q_item_loop.get('marks', 1)

            attempt_time = datetime.now()

            cursor.execute("""
                INSERT INTO test_attempts (student_id, test_id, score, total_marks_possible, 
                                       status, started_at, num_questions_in_attempt) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (student_id, test_id, 0, current_total_marks_possible, 
                  'pending_submission', attempt_time, actual_question_count))
            attempt_id = cursor.lastrowid

            submitted_answers_for_json = {} 

            for question_item_loop in questions: 
                question_id_str = str(question_item_loop['id'])
                selected_option = request.form.get(f'question_{question_id_str}')
                submitted_answers_for_json[question_id_str] = selected_option

                is_correct = False
                question_type = question_item_loop.get('question_type')
                correct_answer_from_db = question_item_loop.get('correct_option')
                question_marks = question_item_loop.get('marks', 1)

                if selected_option is not None and correct_answer_from_db is not None:
                    if question_type == 'mcq':
                        is_correct = selected_option.strip().upper() == correct_answer_from_db.strip().upper()
                    elif question_type == 'fill_blank':
                        is_correct = selected_option.strip().lower() == correct_answer_from_db.strip().lower()
                    elif question_type == 'true_false':
                        is_correct = selected_option == correct_answer_from_db
                
                actual_marks_awarded = 0
                if is_correct:
                    actual_marks_awarded = question_marks
                    current_score += question_marks
                
                cursor.execute("""
                    INSERT INTO student_test_answers (attempt_id, question_id, selected_option, is_correct, marks_awarded)
                    VALUES (%s, %s, %s, %s, %s)
                """, (attempt_id, question_item_loop['id'], selected_option, is_correct, actual_marks_awarded))

            submitted_answers_json_str = json.dumps(submitted_answers_for_json)
            completed_time = datetime.now()

            cursor.execute("""
                UPDATE test_attempts 
                SET score = %s, submitted_answers_json = %s, status = %s, completed_at = %s
                WHERE id = %s
            """, (current_score, submitted_answers_json_str, 'submitted', completed_time, attempt_id))
            
            
            cursor.execute("SELECT level FROM users WHERE id = %s", (student_id,))
            student_user_details = cursor.fetchone()
            current_student_level = student_user_details.get('level') if student_user_details else None

            if current_student_level is None: 
                percentage_score = 0
                if current_total_marks_possible > 0:
                    percentage_score = (current_score / current_total_marks_possible) * 100
                
                new_level = None
                if percentage_score <= 33: new_level = 'Novice'
                elif percentage_score <= 66: new_level = 'Intermediate'
                else: new_level = 'Advance'
                
                if new_level:
                    cursor.execute("UPDATE users SET level = %s WHERE id = %s", (new_level, student_id))
                    logger.info(f"Student ID {student_id} level set to '{new_level}' after test (ID: {test_id}, Score: {percentage_score:.2f}%).")
                    session['level'] = new_level
            
            conn.commit()
            flash(f'Test submitted! Your score: {current_score}/{current_total_marks_possible}. See details below.', 'success')
            return redirect(url_for('view_test_attempt_results', attempt_id=attempt_id))

        return render_template('take_test_form.html', test=test_details, questions=questions)

    except pymysql.Error as db_err:
        logger.error(f"Database error during test taking for test_id {test_id}: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error during test taking for test_id {test_id}: {str(e)}", exc_info=True)
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn: 
            if 'cursor' in locals() and cursor:
                cursor.close()
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/upload-ai-training-material', methods=['POST'])
def upload_ai_training_material():
    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher to upload AI training materials.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        teacher_username = session['username']
        semester = request.form.get('semester')
        subject = request.form.get('subject')
        topic = request.form.get('topic','').strip()
        guidance = request.form.get('guidance', '') 
        file = request.files.get('file')

        material_title = topic 
        material_description_for_db = guidance

        if not all([semester, subject, topic, file and file.filename]):
            flash('Semester, Subject, Topic, and File are required for AI training material.', 'error')
            return redirect(url_for('dashboard'))

        if not file.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed for AI training material.', 'error')
            return redirect(url_for('dashboard'))
            
        teacher_department_info = session.get('department', 'general_uploads') 
        if isinstance(teacher_department_info, str) and ',' in teacher_department_info:
             teacher_department_for_path = teacher_department_info.split(',')[0].strip()
        else:
             teacher_department_for_path = teacher_department_info.strip()

        upload_subfolder_pdf = os.path.join(teacher_department_for_path, subject, 'ai_materials', topic.replace(" ", "_"))
        upload_path_pdf = os.path.join(app.config['UPLOAD_FOLDER'], upload_subfolder_pdf)
        os.makedirs(upload_path_pdf, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = secure_filename(file.filename)
        filename_on_disk = f"{timestamp}_{original_filename}"
        filepath_on_disk_pdf = os.path.join(upload_path_pdf, filename_on_disk)
        
        file.save(filepath_on_disk_pdf)
        stored_filename_path_pdf = os.path.join(upload_subfolder_pdf, filename_on_disk)

       
        try:
            logger.info(f"Starting LangChain processing for PDF: {filepath_on_disk_pdf}")
            loader = PyPDFLoader(filepath_on_disk_pdf)
            documents = loader.load()

            if not documents:
                logger.warning(f"PyPDFLoader returned no documents for {filepath_on_disk_pdf}")
                flash(f"Could not process the PDF content for topic '{topic}'. The PDF might be empty or unreadable.", 'warning')
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                texts = text_splitter.split_documents(documents)

                if not texts:
                    logger.warning(f"Text splitting resulted in no chunks for {filepath_on_disk_pdf}")
                    flash(f"Could not extract text chunks from the PDF for topic '{topic}'.", 'warning')
                else:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                   
                    sane_subject = secure_filename(subject).lower()
                    sane_topic = secure_filename(topic).lower()
                    vector_store_topic_path = os.path.join(VECTOR_STORE_BASE_PATH, sane_subject, sane_topic)

                    
                    if os.path.exists(vector_store_topic_path):
                        shutil.rmtree(vector_store_topic_path)
                    os.makedirs(vector_store_topic_path, exist_ok=True)
                    
                    db = FAISS.from_documents(texts, embeddings)
                    db.save_local(vector_store_topic_path)
                    logger.info(f"FAISS index saved for topic '{topic}' at {vector_store_topic_path}")
                    flash(f"PDF for topic '{topic}' processed and indexed for explanations.", "success")

        except Exception as e_langchain:
            logger.error(f"LangChain processing failed for PDF {filepath_on_disk_pdf}: {str(e_langchain)}", exc_info=True)
            flash(f"Error processing PDF for AI explanations: {str(e_langchain)}", "error")
       

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO materials 
            (semester, subject, title, description, filename, teacher, upload_date, type, topic)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s)
        """, (semester, subject, material_title, material_description_for_db, 
              stored_filename_path_pdf, teacher_username, 'ai_explanation_source', topic))
        conn.commit()
        

    except pymysql.Error as db_err:
        logger.error(f"Database error uploading AI material: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error uploading AI material: {str(e)}", exc_info=True)
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn: conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/get_ai_explanation/<int:question_id>')
def get_ai_explanation(question_id):
    if 'user_id' not in session or session.get('role') != 'student':
        return jsonify({"error": "Unauthorized"}), 401

    logger.info(f"Attempting to get AI explanation for question_id: {question_id}")
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT q.topic, q.question_text, q.correct_option,
                   q.option_a, q.option_b, q.option_c, q.option_d,
                   t.subject as test_subject, t.semester as test_semester
            FROM questions q
            JOIN tests t ON q.test_id = t.id
            WHERE q.id = %s
        """, (question_id,))
        question_info = cursor.fetchone()
       
        if not question_info:
            logger.warning(f"No question_info found for question_id: {question_id}. Returning early.")
            return jsonify({"explanation": "Question details not found.", "topic": "N/A"}), 404

        topic = question_info.get('topic')
        question_text = question_info.get('question_text')
        correct_option_char = question_info.get('correct_option')

        if not topic or not question_text or not correct_option_char:
            logger.warning(f"Missing topic, question_text, or correct_option for question_id: {question_id}.")
            return jsonify({
                "explanation": "Essential information (topic, question text, or correct answer) for this question is missing. Cannot generate explanation.",
                "topic": topic or "N/A"
            }), 200
        
        correct_option_text = ""
        if correct_option_char == 'A': correct_option_text = question_info.get('option_a', '')
        elif correct_option_char == 'B': correct_option_text = question_info.get('option_b', '')
        elif correct_option_char == 'C': correct_option_text = question_info.get('option_c', '')
        elif correct_option_char == 'D': correct_option_text = question_info.get('option_d', '')

        prompt_for_topic_explanation = f"""
You are a friendly tutor. A student is asking for an explanation about a question related to the topic: "{topic}".

Question: "{question_text}"
The correct answer is option: "{correct_option_char}" (which states: "{correct_option_text}").

Please provide a simple and easy-to-understand explanation (3-4 lines) focusing on the core concept of the topic "{topic}" as it relates to this question and why this option is the correct answer.
Do not refer to a student's mistake or their specific answer, as you are providing a general explanation for this question's topic and correct solution.
"""
        try:
           
            client = openai.OpenAI(api_key=openai.api_key) 
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful and patient tutor."},
                    {"role": "user", "content": prompt_for_topic_explanation}
                ],
                max_tokens=150,
                temperature=0.6
            )
            ai_explanation_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed for question_id {question_id}: {str(e)}")
            ai_explanation_text = "Sorry, an explanation could not be generated at this time due to an internal error."

        logger.info(f"Generated AI explanation for question_id {question_id} (topic: {topic}): {ai_explanation_text[:200]}...")
        return jsonify({"explanation": ai_explanation_text, "topic": topic})

   
    except pymysql.Error as db_err:
        logger.error(f"Database error getting AI explanation for question_id {question_id}: {db_err}", exc_info=True)
        return jsonify({"error": f"Database error: {db_err}"}), 500
    except Exception as e:
        logger.error(f"Error getting AI explanation for question_id {question_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}", "topic": "Error"}), 500
    finally:
        if conn:
            conn.close()



@app.route('/api/explanations/<int:attempt_id>')
def get_explanations_for_attempt(attempt_id):
    if 'user_id' not in session or session.get('role') != 'student':
        return jsonify({'error': 'Unauthorized'}), 403

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                q.id as question_id,
                q.question_text,
                q.correct_option AS question_correct_option, 
                q.topic,
                q.option_a, q.option_b, q.option_c, q.option_d,
                q.question_type,
                sta.selected_option AS student_selected_option, 
                sta.is_correct
            FROM student_test_answers sta
            JOIN questions q ON sta.question_id = q.id
            WHERE sta.attempt_id = %s
            ORDER BY q.id ASC
        """, (attempt_id,))
        
        attempt_questions_data = cursor.fetchall()

        processed_questions_for_view = []
        if not attempt_questions_data:
            return jsonify({'questions': [], 'message': 'No answers found for this attempt.'})

        for r_info in attempt_questions_data:
            question_type = r_info.get('question_type')
            student_answer_raw = r_info.get('student_selected_option') 
            correct_answer_from_question_db = r_info.get('question_correct_option')

            student_answer_display_text = ""
            correct_answer_display_text = ""
            mcq_options = None 
            
            options_map = {
                'A': r_info.get('option_a'), 'B': r_info.get('option_b'),
                'C': r_info.get('option_c'), 'D': r_info.get('option_d')
            }

            if question_type == 'mcq':
                student_answer_display_text = options_map.get(student_answer_raw, student_answer_raw if student_answer_raw is not None else "Not Answered")
                correct_answer_display_text = options_map.get(correct_answer_from_question_db)
                mcq_options = { 
                    'A': r_info.get('option_a'),
                    'B': r_info.get('option_b'),
                    'C': r_info.get('option_c'),
                    'D': r_info.get('option_d')
                }
            else: 
                student_answer_display_text = student_answer_raw if student_answer_raw is not None else "Not Answered"
                correct_answer_display_text = correct_answer_from_question_db
            
            ai_explanation_text = "AI explanation placeholder." 

            question_data_to_append = {
                'question_id': r_info['question_id'],
                'question_text': r_info['question_text'],
                'student_answer_text': student_answer_display_text,
                'correct_answer_text': correct_answer_display_text,
                'is_correct': r_info['is_correct'],
                'topic': r_info['topic'],
                'question_type': question_type,
                'explanation': ai_explanation_text
            }
            if mcq_options: 
                question_data_to_append['options'] = mcq_options
            
            processed_questions_for_view.append(question_data_to_append)

        return jsonify({'questions': processed_questions_for_view})

    except pymysql.Error as db_err:
        logger.error(f"Database error in /api/explanations for attempt_id {attempt_id}: {db_err}", exc_info=True)
        return jsonify({'error': f'Database error: {db_err}'}), 500
    except Exception as e:
        logger.error(f"Error in /api/explanations for attempt_id {attempt_id}: {str(e)}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()


@app.route('/test_attempt_results/<int:attempt_id>')
def view_test_attempt_results(attempt_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Please login as a student to view results.', 'error')
        return redirect(url_for('login'))

    google_api_key = "AIzaSyAHCuV_qcx16bkbkmM9-Q6Xrt00ictwlyg" 
    actual_placeholder_key = "AIzaSyBBs-TjqCMAEu74rW6Ahjdcn04bhipRnNc" 

    ai_explanations_available = True 
    if not google_api_key or google_api_key in actual_placeholder_key:
        logger.error("GOOGLE_API_KEY not found or is placeholder for /test_attempt_results route. AI explanations may be limited.")
        ai_explanations_available = False 

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                ta.id AS attempt_id, ta.test_id, ta.student_id, ta.score, 
                ta.total_marks_possible, ta.status AS attempt_status, 
                ta.started_at, ta.completed_at, ta.submitted_answers_json,
                ta.num_questions_in_attempt,
                t.title AS test_title, t.subject AS test_subject, 
                t.semester AS test_semester, t.teacher_username,
                u.username as student_name, u.roll_number, # Added student_name and roll_number here
                (SELECT SUM(q_sum.marks) FROM questions q_sum WHERE q_sum.test_id = ta.test_id) AS actual_test_total_marks
            FROM test_attempts ta
            JOIN tests t ON ta.test_id = t.id
            JOIN users u ON ta.student_id = u.id # Join with users to get student_name
            WHERE ta.id = %s AND ta.student_id = %s
        """, (attempt_id, session['user_id']))
        attempt_details = cursor.fetchone() 

        if not attempt_details:
            flash('Test attempt not found or you are not authorized to view it.', 'error')
            if conn: conn.close()
            return redirect(url_for('dashboard'))
        
        logger.info(f"Fetched attempt_details for attempt ID {attempt_id}: {attempt_details}")

       
        student_percentage = None
        student_level_display = 'N/A'
        score = attempt_details.get('score')
        
        denominator_for_percentage = attempt_details.get('total_marks_possible')
        if not denominator_for_percentage or denominator_for_percentage <= 0:
            denominator_for_percentage = attempt_details.get('actual_test_total_marks')
            if not (denominator_for_percentage and denominator_for_percentage > 0) and \
               attempt_details.get('num_questions_in_attempt') and attempt_details['num_questions_in_attempt'] > 0:
                denominator_for_percentage = attempt_details['num_questions_in_attempt']
        
        if denominator_for_percentage is not None and denominator_for_percentage > 0 and score is not None:
            student_percentage = (float(score) / float(denominator_for_percentage)) * 100
            attempt_details['percentage'] = student_percentage 
            if 0 <= student_percentage <= 33: student_level_display = 'Novice'
            elif student_percentage <= 66: student_level_display = 'Intermediate'
            else: student_level_display = 'Advance'
        else: attempt_details['percentage'] = None
        attempt_details['level_display'] = student_level_display
        
        
        rank = "N/A"; total_participants = 0
        if attempt_details.get('score') is not None and attempt_details.get('started_at') is not None:
            cursor.execute("""
                SELECT student_id, score, started_at FROM test_attempts
                WHERE test_id = %s AND score IS NOT NULL AND started_at IS NOT NULL AND status = 'submitted'
                ORDER BY score DESC, started_at ASC 
            """, (attempt_details['test_id'],))
            all_attempts_for_test = cursor.fetchall()
            total_participants = len(all_attempts_for_test)
            if total_participants > 0:
                rank = sum(1 for oa in all_attempts_for_test if oa['score'] > score or (oa['score'] == score and oa['started_at'] < attempt_details['started_at'])) + 1
        
        
        cursor.execute("""
            SELECT q.id as question_id, q.question_text, q.option_a, q.option_b, q.option_c, q.option_d,
                   q.correct_option, q.topic, q.question_type, q.image_filename, q.marks as question_marks,
                   sta.selected_option, sta.is_correct, sta.marks_awarded
            FROM questions q
            JOIN student_test_answers sta ON q.id = sta.question_id
            WHERE q.test_id = %s AND sta.attempt_id = %s
            ORDER BY q.id ASC
        """, (attempt_details['test_id'], attempt_id))
        questions_raw = cursor.fetchall()

        processed_questions = [] 
        if questions_raw:
            llm_for_explanation = None 
            if ai_explanations_available:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm_for_explanation = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-latest", 
                        google_api_key=google_api_key,
                        temperature=0.5,
                        convert_system_message_to_human=True
                    )
                except Exception as e_llm_init:
                    logger.error(f"Failed to initialize ChatGoogleGenerativeAI for explanations: {e_llm_init}")
                    ai_explanations_available = False 

            for q_raw_item in questions_raw: 
                qa_item = dict(q_raw_item) 
                
                if qa_item.get('image_filename'):
                    qa_item['image_filename'] = str(qa_item['image_filename']).replace('\\', '/')

                question_type = qa_item.get('question_type')
                options_map = {'A': qa_item.get('option_a'), 'B': qa_item.get('option_b'), 'C': qa_item.get('option_c'), 'D': qa_item.get('option_d')}
                if question_type == 'mcq':
                    qa_item['selected_option_text'] = options_map.get(str(qa_item.get('selected_option')).strip().upper(), qa_item.get('selected_option'))
                    qa_item['correct_option_text'] = options_map.get(str(qa_item.get('correct_option')).strip().upper())
                else: 
                    qa_item['selected_option_text'] = qa_item.get('selected_option')
                    qa_item['correct_option_text'] = qa_item.get('correct_option')
                
                
                if ai_explanations_available and llm_for_explanation: 
                    qa_item['langchain_explanation_html'] = generate_general_ai_explanation_for_question(
                        question_details=qa_item,
                        subject_name=attempt_details['test_subject'],
                        google_api_key=google_api_key
                    )
                elif not ai_explanations_available:
                    qa_item['langchain_explanation_html'] = "<p><small>AI explanations are unavailable (API key issue).</small></p>"
                else: 
                    qa_item['langchain_explanation_html'] = "<p><small>AI explanation could not be generated (LLM init failed).</small></p>"
                
                processed_questions.append(qa_item)
        
        questions_with_answers = processed_questions
        
        logger.info(f"--- Data for student_test_results (attempt_id {attempt_id}) ---")
        
        return render_template(
            'student_test_results.html',
            attempt=attempt_details, 
            questions_with_answers=questions_with_answers,
            rank=rank, 
            total_participants=total_participants
        )

    except pymysql.Error as db_err:
        logger.error(f"Database error viewing test attempt results for attempt_id {attempt_id}: {db_err}", exc_info=True)
        flash(f'Database error: {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error viewing test attempt results for attempt_id {attempt_id}: {str(e)}", exc_info=True)
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    finally:
        if conn: 
            if 'cursor' in locals() and cursor: cursor.close()
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/view-attempt/<int:attempt_id>')
def view_attempt_page(attempt_id):
    if 'user_id' not in session or session['role'] != 'student':
        flash("Please log in to view your attempts", "error")
        return redirect('/login')
    return render_template("view_attempt.html", attempt_id=attempt_id)


def generate_general_ai_explanation_for_question(question_details, subject_name, google_api_key):
    if not google_api_key or google_api_key == "AIzaSyAHCuV_qcx16bkbkmM9-Q6Xrt00ictwlyg": 
        logger.warning("Google API Key not provided or is placeholder for general explanation.")
        return "<p><small>AI explanations are currently unavailable (API key issue).</small></p>"

    question_id = question_details.get('id') 
    topic = question_details.get('topic', '')
    question_text = question_details.get('question_text', '')
    correct_option_char_or_answer = question_details.get('correct_option', '') 
    question_type = question_details.get('question_type') 
    options_map = {
        'A': question_details.get('option_a'), 'B': question_details.get('option_b'),
        'C': question_details.get('option_c'), 'D': question_details.get('option_d')
    }
    
    correct_answer_display_text = ""
    if question_type == 'mcq':
        correct_answer_display_text = options_map.get(correct_option_char_or_answer, '')
    else:
        correct_answer_display_text = correct_option_char_or_answer

   
    if not all([topic, question_text, correct_option_char_or_answer]):
        logger.warning(f"Missing essential info (topic, text, or correct option) for general explanation of Q_ID {question_id}, Type: {question_type}")
        return "<p><small>Could not generate explanation due to missing core question details.</small></p>"
    if question_type == 'mcq' and not correct_answer_display_text:
        logger.warning(f"Missing correct_answer_display_text for MCQ Q_ID {question_id}")
        return "<p><small>Could not generate explanation due to missing MCQ answer text.</small></p>"


    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                     google_api_key=google_api_key,
                                     temperature=0.5, 
                                     convert_system_message_to_human=True)

    
        text_to_check_for_coding = (topic.lower() + " " + subject_name.lower())
        is_code_question = any(keyword in text_to_check_for_coding for keyword in CODING_KEYWORDS)

        prompt_parts = [
            "You are a helpful and patient tutor. Please provide a detailed and easy-to-understand explanation (around 3-5 sentences, or more if needed for clarity, aiming for comprehensive understanding) for the following question and its correct answer."
        ]

        if is_code_question:
            prompt_parts.append("The question is related to a programming/coding concept.")
            prompt_parts.append("If applicable, include a concise and clear code snippet (e.g., in Python or pseudocode, enclosed in ```python ... ``` or ``` ... ``` for pseudocode) to illustrate the core concept being explained. The code should be minimal, directly relevant, and help clarify the explanation.")
        else:
            prompt_parts.append("If applicable, include a simple and relatable real-world example to help illustrate the concept.")
        
        
        prompt_parts.append(f"\nQuestion: \"{question_text}\"")

        if question_type == 'mcq':
            prompt_parts.append(f"The correct answer is option {correct_option_char_or_answer}: \"{correct_answer_display_text}\".")
        elif question_type == 'true_false':
            prompt_parts.append(f"The correct answer is: \"{correct_answer_display_text}\".")
        elif question_type == 'fill_blank':
            prompt_parts.append(f"The correct answer for the blank is: \"{correct_answer_display_text}\".")
        else: 
            prompt_parts.append(f"The correct answer is: \"{correct_answer_display_text}\".")

        prompt_parts.append(f"\nExplain in detail why this is the correct answer, focusing on the topic: '{topic}'. Elaborate on the underlying principles and concepts to ensure the student understands thoroughly.")
        prompt_parts.append("Structure your explanation clearly. If providing a code snippet, briefly explain what the code does and how it relates to the concept.")

        final_prompt = "\n".join(prompt_parts)
        logger.info(f"General Explanation Prompt for Q_ID {question_id} (Type: {question_type}, Code-related: {is_code_question}): {final_prompt[:500]}...") # Log more of the prompt

        ai_response = llm.invoke(final_prompt)
        explanation = ai_response.content.strip()
        
        formatted_explanation = explanation.replace("```python", "<pre><code class='language-python'>").replace("```plaintext", "<pre><code>").replace("```", "</code></pre>")
       
        parts = formatted_explanation.split("</pre>")
        processed_parts = []
        for i, part in enumerate(parts):
            if i < len(parts) -1 : 
                code_block_match = part.rfind("<pre><code")
                if code_block_match != -1:
                    code_content = part[code_block_match:]
                    non_code_content = part[:code_block_match].replace("\n\n", "</p><p>").replace("\n", "<br>")
                    processed_parts.append(non_code_content + code_content + "</pre>")
                else: 
                     processed_parts.append(part.replace("\n\n", "</p><p>").replace("\n", "<br>") + "</pre>")
            else: 
                processed_parts.append(part.replace("\n\n", "</p><p>").replace("\n", "<br>"))
        
        final_html_explanation = "<p>" + "".join(processed_parts) + "</p>"
        final_html_explanation = final_html_explanation.replace("<p><br></p>", "").replace("<p></p>", "")

        return final_html_explanation

    except Exception as e_llm:
        logger.error(f"General LLM explanation failed for Q_ID {question_id}, topic '{topic}', type '{question_type}': {e_llm}", exc_info=True)
        if "API key not valid" in str(e_llm) or "PERMISSION_DENIED" in str(e_llm):
            return "<p><small>Could not generate explanation: Issue with Google API key/permissions.</small></p>"
        elif "quota" in str(e_llm).lower() or "rate_limit_exceeded" in str(e_llm).lower():
            return "<p><small>Could not generate explanation: API usage limit reached. Check Google Cloud/AI Studio quota.</small></p>"
        return "<p><small>An error occurred while generating the AI explanation for this question.</small></p>"


@app.route('/test/<int:test_id>/review_with_explanations')
def view_test_review_with_explanations(test_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Please login as a student to view this page.', 'error')
        return redirect(url_for('login'))

    google_api_key = "AIzaSyBBs-TjqCMAEu74rW6Ahjdcn04bhipRnNc" 
    
    if not google_api_key or google_api_key == "AIzaSyAHCuV_qcx16bkbkmM9-Q6Xrt00ictwlyg":
        logger.error("GOOGLE_API_KEY not found or not replaced for /test/.../review_with_explanations route.")
        flash("AI explanations are currently unavailable due to a configuration issue.", "warning")
        pass 

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, title, subject, semester FROM tests WHERE id = %s", (test_id,))
        test_details = cursor.fetchone()

        if not test_details:
            flash('Test not found.', 'error')
            return redirect(url_for('dashboard'))

        cursor.execute("""
            SELECT id, question_text, option_a, option_b, option_c, option_d, 
                   correct_option, topic, question_type, image_filename, marks
            FROM questions
            WHERE test_id = %s
            ORDER BY id ASC
        """, (test_id,))
        questions_raw = cursor.fetchall()

        questions_with_explanations = []
        if questions_raw:
            for q_raw in questions_raw:
                q_detail = dict(q_raw)
                
                if q_detail.get('image_filename'):
                    q_detail['image_filename'] = str(q_detail['image_filename']).replace('\\', '/')
                    logger.info(f"ReviewExplanations - Normalized image_filename for Q_ID {q_detail['id']}: {q_detail['image_filename']}")
                
                q_detail['ai_explanation_html'] = generate_general_ai_explanation_for_question(
                    q_detail, 
                    test_details['subject'], 
                    google_api_key 
                )
                questions_with_explanations.append(q_detail)
        
        # Log data being passed to template for debugging
        logger.info(f"--- Questions data for review_with_explanations (test_id {test_id}): ---")
        for q_idx, q_data in enumerate(questions_with_explanations):
            logger.info(f"Review Question {q_idx + 1} Image: {q_data.get('image_filename')}") # Check this log
        logger.info("--- End of review questions data ---")

        return render_template(
            'test_review_explanations.html',
            test=test_details,
            questions_data=questions_with_explanations
        )

    except pymysql.Error as db_err:
        logger.error(f"Database error in view_test_review_with_explanations for test_id {test_id}: {db_err}", exc_info=True)
        flash('A database error occurred while preparing the test review.', 'error')
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Error in view_test_review_with_explanations for test_id {test_id}: {e}", exc_info=True)
        flash('An unexpected error occurred while preparing the test review.', 'error')
        return redirect(url_for('dashboard'))
    finally:
        if conn: 
            conn.close()



@app.route('/test_route_check')
def test_route_check():
    return "Test route is working!"

@app.route('/admin/add_teacher', methods=['POST'])
def admin_add_teacher():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('Username, email, and password are required.', 'error')
            return redirect(url_for('dashboard'))

        hashed_password = hash_password(password)
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username=%s OR email=%s", (username, email))
            if cursor.fetchone():
                flash('Teacher with this username or email already exists.', 'error')
            else:
                cursor.execute("""
                    INSERT INTO users (username, email, password, role, status)
                    VALUES (%s, %s, %s, 'teacher', 'approved')
                """, (username, email, hashed_password))
                conn.commit()
                flash('Teacher added successfully and approved!', 'success')
        except pymysql.Error as db_err:
            logger.error(f"Database error adding teacher: {db_err}", exc_info=True)
            flash(f'Database error: {db_err}', 'error')
            if conn: conn.rollback()
        except Exception as e:
            logger.error(f"Error adding teacher: {e}", exc_info=True)
            flash(f'An unexpected error occurred: {e}', 'error')
            if conn: conn.rollback()
        finally:
            if conn: conn.close()
    return redirect(url_for('dashboard'))


@app.route('/admin/remove_teacher/<int:user_id>', methods=['POST'])
def admin_remove_teacher(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE id = %s", (user_id,))
        user_to_remove = cursor.fetchone()
        if user_to_remove and user_to_remove['role'] == 'teacher':
            cursor.execute("DELETE FROM users WHERE id = %s AND role = 'teacher'", (user_id,))
            conn.commit()
            if cursor.rowcount > 0:
                flash('Teacher removed successfully!', 'success')
            else:
                flash('Teacher not found or already removed.', 'warning')
        else:
            flash('User is not a teacher or does not exist.', 'error')
    except pymysql.Error as db_err:
        logger.error(f"Database error removing teacher: {db_err}", exc_info=True)
        flash(f'Database error: {db_err}', 'error')
        if conn: conn.rollback()
    except Exception as e:
        logger.error(f"Error removing teacher: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'error')
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return redirect(url_for('dashboard'))

@app.route('/admin/approve_teacher/<int:teacher_id>', methods=['POST'])
def admin_approve_teacher(teacher_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET status = 'approved' 
            WHERE id = %s AND role = 'teacher' AND status = 'pending'
        """, (teacher_id,))
        conn.commit()
        if cursor.rowcount > 0:
            flash('Teacher account approved successfully!', 'success')
        else:
            flash('Teacher not found, already approved, or not a pending teacher.', 'warning')
    except pymysql.Error as db_err:
        logger.error(f"Database error approving teacher: {db_err}", exc_info=True)
        flash(f'Database error: {db_err}', 'error')
        if conn: conn.rollback()
    except Exception as e:
        logger.error(f"Error approving teacher: {e}", exc_info=True)
        flash(f'An unexpected error occurred: {e}', 'error')
        if conn: conn.rollback()
    finally:
        if conn: conn.close()
    return redirect(url_for('dashboard'))


@app.route('/admin/add_course_entry', methods=['POST'])
def admin_add_course_entry():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Unauthorized access.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        semester = request.form.get('semester')
        department = request.form.get('department')
        subject_name = request.form.get('subject_name')

        if not semester or not department or not subject_name:
            flash('Semester, Department, and Subject Name are required.', 'error')
            return redirect(url_for('dashboard'))

        excel_path = os.path.join(os.path.dirname(__file__), 'CSE_Semester_Subjects_Credits.xlsx')

        try:
            if not os.path.exists(excel_path):
                df_new = pd.DataFrame(columns=['Semester', 'Department', 'Subject'])
                df_new.to_excel(excel_path, index=False)
                logger.info(f"Created new Excel file with headers at: {excel_path}")

            df = pd.read_excel(excel_path)
            new_entry_data = {
                'Semester': [semester],
                'Department': [department],
                'Subject': [subject_name]
            }
            new_entry = pd.DataFrame(new_entry_data)

            df_normalized_subject = df['Subject'].astype(str).str.strip().str.lower()
            df_normalized_semester = df['Semester'].astype(str).str.strip().str.lower() 
            df_normalized_department = df['Department'].astype(str).str.strip().str.lower()

            normalized_new_subject = str(subject_name).strip().lower()
            normalized_new_semester = str(semester).strip().lower()
            normalized_new_department = str(department).strip().lower()
            
            is_duplicate = (
                (df_normalized_subject == normalized_new_subject) &
                (df_normalized_semester == normalized_new_semester) &
                (df_normalized_department == normalized_new_department)
            ).any()

            if is_duplicate:
                flash('This course entry (Semester, Department, Subject) already exists in the Excel file.', 'warning')
            else:
                df_updated = pd.concat([df, new_entry], ignore_index=True)
                df_updated.to_excel(excel_path, index=False)
                flash('New course entry added to Excel successfully!', 'success')

        except FileNotFoundError:
            logger.error(f"Excel file not found at: {excel_path}. Attempted to create it but failed or an error occurred before write.")
            flash(f'Error: The Excel file was not found and could not be created. Please check server permissions and path.', 'error')
        except PermissionError: 
            logger.error(f"Permission denied when trying to write to Excel file: {excel_path}", exc_info=True)
            flash('Error: Permission denied. Could not write to the Excel file.', 'error')
        except Exception as e:
            logger.error(f"Error adding course entry to Excel: {e}", exc_info=True)
            flash(f'An unexpected error occurred while updating Excel: {e}', 'error')

    return redirect(url_for('dashboard'))


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE reset_token = %s AND reset_token_expires > %s",
                       (token, datetime.now()))
        user = cursor.fetchone()

        if not user:
            flash('The password reset link is invalid or has expired.', 'error')
            return redirect(url_for('forgot_password'))
      

        if request.method == 'POST':
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not new_password or not confirm_password:
                 flash('Please enter and confirm your new password.', 'error')
               
                 return render_template('reset_password.html', token=token)

            if new_password != confirm_password:
                flash('Passwords do not match.', 'error')
               
                return render_template('reset_password.html', token=token)

            
            hashed_password = hash_password(new_password) 

            cursor.execute("""
                UPDATE users 
                SET password = %s, reset_token = NULL, reset_token_expires = NULL 
                WHERE id = %s
            """, (hashed_password, user['id']))
            conn.commit()
           

            flash('Your password has been reset successfully! Please login.', 'success')
            return redirect(url_for('login'))

        
        return render_template('reset_password.html', token=token)

    except Exception as e:
        logger.error(f"Reset password error: {str(e)}", exc_info=True)
        flash('An error occurred during the password reset process. Please try again.', 'error')
        return redirect(url_for('forgot_password'))
    finally:
        if conn:
            conn.close()


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email_or_username = request.form.get('email_or_username')
        if not email_or_username:
            flash('Please enter your email or username.', 'error')
            return redirect(url_for('forgot_password'))

        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

           
            cursor.execute("SELECT id, email, username FROM users WHERE email = %s OR username = %s", 
                           (email_or_username, email_or_username))
            user = cursor.fetchone()

            if user:
                token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(hours=1)

                cursor.execute("""
                    UPDATE users 
                    SET reset_token = %s, reset_token_expires = %s 
                    WHERE id = %s
                """, (token, expires_at, user['id']))
                conn.commit()

                
                reset_link = url_for('reset_password', token=token, _external=True)
                email_body = f"""
                Hello {user['username']},

                Someone requested a password reset for your Smart Tutoring System account.
                If this was you, please click the link below to reset your password:
                {reset_link}

                This link will expire in 1 hour.

                If you did not request a password reset, please ignore this email.
                Your password will remain unchanged.

                Thank you,
                The Smart Tutoring System Team
                """
                email_subject = "Password Reset Request - Smart Tutoring System"
                receiver_email = user['email']
                
                logger.info(f"Password reset requested for user: {user['username']} (ID: {user['id']}). Token: {token}")
                logger.info(f"Reset link (for testing/manual use if email fails): {reset_link}")
            
            
            flash('If an account with that email or username exists, a password reset link has been sent. Please check your email.', 'info')
            return redirect(url_for('login')) 

        except pymysql.Error as db_err:
            logger.error(f"Database error during forgot password: {db_err}", exc_info=True)
            flash('A database error occurred. Please try again.', 'error')
        except Exception as e:
            logger.error(f"Forgot password error: {str(e)}", exc_info=True)
            flash('An error occurred. Please try again.', 'error')
        finally:
            if conn:
                conn.close()
       
        return redirect(url_for('forgot_password')) 

    return render_template('forgot_password.html')



@app.route('/start-live-lecture', methods=['POST'])
def start_live_lecture():
    if 'user_id' not in session or session.get('role') != 'teacher':
        flash('Please login as a teacher to start a lecture.', 'error')
        return redirect(url_for('login'))

    teacher_id = session['user_id']
    teacher_username = session['username']
    semester = request.form.get('semester')
    subject = request.form.get('subject')
    title = request.form.get('title') 

 
    if not all([semester, subject]): 
        flash('Semester and Subject are required to start a lecture.', 'error')
        return redirect(url_for('dashboard'))

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()


        cursor.execute("""
            INSERT INTO live_lectures (teacher_id, teacher_username, subject, semester, title, status, started_at, meeting_link)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (teacher_id, teacher_username, subject, semester, title, 'live', datetime.now(), "#internal")) 
        
        conn.commit()
        lecture_id = cursor.lastrowid 

        
        internal_meeting_link = url_for('lecture_room', lecture_id=lecture_id, _external=False) 
        cursor.execute("UPDATE live_lectures SET meeting_link = %s WHERE id = %s", (internal_meeting_link, lecture_id))
        conn.commit()

        flash(f'Live lecture "{title if title else subject}" (ID: {lecture_id}) started successfully!', 'success')
        logger.info(f"Teacher {teacher_username} (ID: {teacher_id}) started internal live lecture ID {lecture_id} for {subject}, Sem {semester}.")
        
        return redirect(url_for('lecture_room', lecture_id=lecture_id))
        
    except pymysql.Error as db_err:
        logger.error(f"Database error starting live lecture: {db_err}", exc_info=True)
        if conn: conn.rollback()
        flash(f'Database error: Could not start live lecture. {db_err}', 'error')
    except Exception as e:
        logger.error(f"Error starting live lecture: {e}", exc_info=True)
        if conn: conn.rollback()
        flash(f'An unexpected error occurred: {e}', 'error')
    finally:
        if conn:
            conn.close()
    
    return redirect(url_for('dashboard'))


@app.route('/calendar')
def calendar():
    if 'username' not in session:
        return redirect('/login')
    return render_template('calendar.html')

@app.route('/lecture/<string:lecture_id>')
def lecture_room(lecture_id):
    if 'user_id' not in session:
        flash('Please login to join a lecture.', 'error')
        return redirect(url_for('login'))
    
    username = session.get('username', 'Anonymous')
    user_role = session.get('role', 'student') 
    
    return render_template('lecture_room.html', 
                           lecture_id=lecture_id, 
                           username=username,
                           user_role=user_role) 

@socketio.on('connect')
def handle_connect():
    user_id = session.get('user_id')
    username = session.get('username', 'Unknown User')
    if user_id:
        logger.info(f"Socket.IO Client connected: SID {request.sid}, User: {username} (ID: {user_id})")
    else:
        logger.info(f"Socket.IO Client connected: SID {request.sid} (User not logged in or session missing)")


@socketio.on('join_lecture')
def handle_join_lecture(data):
    username = session.get('username', 'Anonymous')
    lecture_id = data.get('lecture_id')
    if not lecture_id:
        logger.warning(f"User {username} (SID: {request.sid}) tried to join lecture with no lecture_id.")
        return

    join_room(lecture_id) 
    logger.info(f"User {username} (SID: {request.sid}) joined lecture room: {lecture_id}")
    socketio.emit('user_joined', {'username': username, 'sid': request.sid}, room=lecture_id, include_self=False)


@socketio.on('leave_lecture')
def handle_leave_lecture(data):
    username = session.get('username', 'Anonymous')
    lecture_id = data.get('lecture_id')
    if not lecture_id:
        logger.warning(f"User {username} (SID: {request.sid}) tried to leave lecture with no lecture_id.")
        return

    leave_room(lecture_id)
    logger.info(f"User {username} (SID: {request.sid}) left lecture room: {lecture_id}")
    socketio.emit('user_left', {'username': username, 'sid': request.sid}, room=lecture_id, include_self=False)


@socketio.on('disconnect')
def handle_disconnect():
    username = session.get('username', 'Anonymous User')
    logger.info(f"Socket.IO Client disconnected: SID {request.sid}, User: {username}")


@socketio.on('lecture_message')
def handle_lecture_message(data):
    lecture_id = data.get('lecture_id')
    message = data.get('message')
    sender = data.get('sender', 'Anonymous') 

    logger.info(f"[SocketIO] 'lecture_message' received. Data: {data}")

    if not all([lecture_id, message, sender != 'Anonymous']): 
        logger.warning(f"[SocketIO] Missing lecture_id, message, or valid sender in 'lecture_message' event. Data: {data}")
        return 

    logger.info(f"[SocketIO] Message in room '{lecture_id}' from '{sender}': '{message}'")
    
    try:
        socketio.emit('new_message', 
                      {'message': message, 'sender': sender}, 
                      room=lecture_id, 
                      include_self=False)
        logger.info(f"[SocketIO] Broadcasted 'new_message' to room '{lecture_id}' (excluding sender '{sender}').")
    except Exception as e:
        logger.error(f"[SocketIO] Error broadcasting 'new_message' to room '{lecture_id}': {e}", exc_info=True)


@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    lecture_id = data.get('lecture_id')
    offer = data.get('offer')
    target_sid = data.get('target_sid') 
    sender_sid = request.sid 
    sender_username = session.get('username', 'Unknown')

    if not all([lecture_id, offer, target_sid]):
        logger.error(f"WEBRTC_OFFER: Missing data from {sender_username} (SID: {sender_sid}) for target {target_sid}")
        return

    logger.info(f"WEBRTC_OFFER: from {sender_username} (SID: {sender_sid}) to SID {target_sid} in room {lecture_id}")
    
    socketio.emit('offer_received', {
        'offer': offer, 
        'sender_sid': sender_sid,
        'sender_username': sender_username
    }, room=target_sid)

@socketio.on('webrtc_answer')
def handle_webrtc_answer(data):
    lecture_id = data.get('lecture_id')
    answer = data.get('answer')
    target_sid = data.get('target_sid') 
    sender_sid = request.sid 
    sender_username = session.get('username', 'Unknown')

    if not all([lecture_id, answer, target_sid]):
        logger.error(f"WEBRTC_ANSWER: Missing data from {sender_username} (SID: {sender_sid}) for target {target_sid}")
        return

    logger.info(f"WEBRTC_ANSWER: from {sender_username} (SID: {sender_sid}) to SID {target_sid} in room {lecture_id}")
    
    socketio.emit('answer_received', {
        'answer': answer, 
        'sender_sid': sender_sid,
        'sender_username': sender_username
    }, room=target_sid)

@socketio.on('webrtc_ice_candidate')
def handle_webrtc_ice_candidate(data):
    lecture_id = data.get('lecture_id')
    candidate = data.get('candidate')
    target_sid = data.get('target_sid') 
    sender_sid = request.sid 
    sender_username = session.get('username', 'Unknown')

    if not all([lecture_id, candidate, target_sid]):
        logger.error(f"WEBRTC_ICE_CANDIDATE: Missing data from {sender_username} (SID: {sender_sid}) for target {target_sid}")
        return
        
    logger.info(f"WEBRTC_ICE_CANDIDATE: Relaying from SID {request.sid} to SID {data.get('target_sid')}. Candidate is present: {data.get('candidate') is not None}")
    socketio.emit('ice_candidate_received', {
        'candidate': data.get('candidate'),
        'sender_sid': request.sid
    }, room=data.get('target_sid'))


@socketio.on('join_lecture')
def handle_join_lecture(data):
    username = session.get('username', 'Anonymous')
    user_role = session.get('role', 'student') 
    lecture_id = data.get('lecture_id')

    if not lecture_id:
        logger.warning(f"User {username} (SID: {request.sid}) tried to join lecture with no lecture_id.")
        return

    join_room(lecture_id)
    logger.info(f"User {username} (Role: {user_role}, SID: {request.sid}) joined lecture room: {lecture_id}")

    
    socketio.emit('user_joined', {
        'username': username, 
        'sid': request.sid,
        'role': user_role 
    }, room=lecture_id, skip_sid=request.sid) 


@app.route('/end-live-lecture/<int:lecture_id>', methods=['POST']) 
def end_live_lecture(lecture_id):
        if 'user_id' not in session or session.get('role') != 'teacher':
            flash('Unauthorized action.', 'error')
            logger.warning(f"Unauthorized attempt to end lecture {lecture_id} by user {session.get('username', 'Unknown')}")
    
            return redirect(url_for('login')) 

        logger.info(f"Attempting to end lecture ID: {lecture_id} by teacher ID: {session['user_id']}")
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            update_query = """
                UPDATE live_lectures 
                SET status = 'ended', ended_at = %s 
                WHERE id = %s AND teacher_id = %s AND status = 'live'
            """
            current_time = datetime.now()
            
            cursor.execute(update_query, (current_time, lecture_id, session['user_id']))
            conn.commit()

            if cursor.rowcount > 0:
                flash('Live lecture has been successfully ended.', 'success') 
                logger.info(f"Lecture ID {lecture_id} status updated to 'ended' by teacher ID {session['user_id']}.")
                if 'socketio' in globals() or 'socketio' in locals():
                    socketio.emit('lecture_ended', {'lecture_id': str(lecture_id)}, room=str(lecture_id))
                
                return jsonify({'status': 'success', 'message': 'Lecture ended successfully'}), 200
            else:
                flash('Could not end lecture. It may have already ended or was not found.', 'warning')
                logger.warning(f"Failed to end lecture ID {lecture_id} for teacher ID {session['user_id']}. Rowcount: {cursor.rowcount}.")
                return jsonify({'status': 'error', 'message': 'Could not end lecture. Already ended or not found.'}), 404
        
        except pymysql.Error as db_err:
            logger.error(f"Database error ending lecture {lecture_id}: {db_err}", exc_info=True)
            flash('Database error: Could not end lecture.', 'error')
            return jsonify({'status': 'error', 'message': 'Database error.'}), 500
        except Exception as e:
            logger.error(f"Error ending lecture {lecture_id}: {e}", exc_info=True)
            flash('An unexpected error occurred while ending the lecture.', 'error')
            return jsonify({'status': 'error', 'message': 'Unexpected server error.'}), 500
        finally:
            if conn:
                conn.close()
        
        return redirect(url_for('dashboard'))




@app.route('/start_eye_control', methods=['POST']) 
def start_eye_control():
    global eye_tracker_process
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in.'}), 401

    if session.get('eye_control_active', False) and eye_tracker_process and eye_tracker_process.poll() is None:
        logger.info("Eye control start requested, but process seems already active.")
        return jsonify({'status': 'info', 'message': 'Eye control is already active.'})

    try:
        script_path = os.path.join(os.path.dirname(__file__), 'eye_mouse_tracker.py')
        if not os.path.exists(script_path):
            logger.error(f"Eye tracking script not found: {script_path}")
            return jsonify({'status': 'error', 'message': f'Eye tracking script not found at {script_path}.'}), 500

        if eye_tracker_process and eye_tracker_process.poll() is None:
            logger.warning("Found an existing eye_tracker_process that was not None. Attempting to terminate before starting new.")
            try:
                eye_tracker_process.terminate()
                eye_tracker_process.wait(timeout=2)
            except Exception as e_term:
                logger.error(f"Error terminating lingering eye_tracker_process: {e_term}")
        
        eye_tracker_process = subprocess.Popen([sys.executable, script_path])
        session['eye_control_active'] = True
        session['initiate_eye_tracking_on_load'] = False 
        logger.info(f"User {session.get('username')} started eye control. Process ID: {eye_tracker_process.pid}")
        return jsonify({'status': 'success', 'message': 'Eye control initiated. OpenCV window should appear. Press "q" in that window to stop.'})

    except Exception as e:
        logger.error(f"Error starting eye control: {e}", exc_info=True)
        session['eye_control_active'] = False
        session['initiate_eye_tracking_on_load'] = False
        return jsonify({'status': 'error', 'message': f'Error starting eye control: {str(e)}'}), 500


# In app.py

@app.route('/stop_eye_control', methods=['POST']) 
def stop_eye_control():
    global eye_tracker_process
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in.'}), 401

    if eye_tracker_process and eye_tracker_process.poll() is None: 
        try:
            eye_tracker_process.terminate()
            eye_tracker_process.wait(timeout=5) 
            logger.info(f"Attempted to terminate eye tracker process {eye_tracker_process.pid}.")
            if eye_tracker_process.poll() is None: 
                 logger.warning(f"Eye tracker process {eye_tracker_process.pid} did not terminate after 5s. May need to be killed manually or use .kill().")
                 
        except subprocess.TimeoutExpired:
            logger.warning(f"Eye tracker process {eye_tracker_process.pid} did not terminate in time on wait().")
        except Exception as e:
            logger.error(f"Error terminating eye tracker process: {e}", exc_info=True)
        eye_tracker_process = None 
    else:
        logger.info("Stop eye control called, but no active process found or process already terminated.")
        eye_tracker_process = None

    session['eye_control_active'] = False
    session['initiate_eye_tracking_on_load'] = False 
    return jsonify({'status': 'success', 'message': 'Eye control has been stopped.'})

@app.route('/web_eye_tracker')
def web_eye_tracker_page():
    if 'user_id' not in session:
        flash('Please login to use this feature.', 'error')
        return redirect(url_for('login'))
    
    if 'initiate_web_eye_tracking_on_load' in session:
        session.pop('initiate_web_eye_tracking_on_load', None)
        logger.info("Cleared 'initiate_web_eye_tracking_on_load' session flag for user.")

    return render_template('web_eye_tracker.html')


@app.route('/download_report_card/<int:attempt_id>')
def download_report_card(attempt_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Unauthorized access. Please login as a student.', 'error')
        return redirect(url_for('login'))

    conn = None
    try:
        
        from weasyprint import HTML
    except ImportError as e_weasy:
        logger.error(f"WeasyPrint import error during PDF generation for report card (attempt {attempt_id}): {e_weasy}")
        logger.error("WeasyPrint library is not installed or its dependencies are missing. Please check installation.")
        flash("PDF generation service is unavailable due to a missing library. Please contact support.", "error")
        return redirect(url_for('view_test_attempt_results', attempt_id=attempt_id))
    except OSError as e_os_weasy: 
        logger.error(f"WeasyPrint OSError (missing C libraries) for report card (attempt {attempt_id}): {e_os_weasy}")
        logger.error("WeasyPrint C dependencies (like GTK3) are missing or not configured correctly. Please check installation.")
        flash("PDF generation failed: Essential system libraries for PDF creation are missing. Please contact support.", "error")
        return redirect(url_for('view_test_attempt_results', attempt_id=attempt_id))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

       
        cursor.execute("""
            SELECT 
                ta.id AS attempt_id, ta.test_id, ta.student_id, ta.score, 
                ta.total_marks_possible, ta.status AS attempt_status, 
                ta.started_at, ta.completed_at, ta.num_questions_in_attempt,
                t.title AS test_title, t.subject AS test_subject, 
                t.semester AS test_semester, t.teacher_username,
                u.username AS student_name, u.roll_number,
                (SELECT SUM(q_sum.marks) FROM questions q_sum WHERE q_sum.test_id = ta.test_id) AS actual_test_total_marks
            FROM test_attempts ta
            JOIN tests t ON ta.test_id = t.id
            JOIN users u ON ta.student_id = u.id
            WHERE ta.id = %s AND ta.student_id = %s
        """, (attempt_id, session['user_id']))
        attempt_details = cursor.fetchone()

        if not attempt_details:
            flash('Test attempt not found or you are not authorized to view it.', 'error')
            if conn: conn.close()
            return redirect(url_for('dashboard'))

       
        student_percentage = None
        student_level_display = 'N/A'
        score = attempt_details.get('score')
        denominator_for_percentage = attempt_details.get('total_marks_possible')
        if not denominator_for_percentage or denominator_for_percentage <= 0:
            denominator_for_percentage = attempt_details.get('actual_test_total_marks')
            if not (denominator_for_percentage and denominator_for_percentage > 0) and \
               attempt_details.get('num_questions_in_attempt') and attempt_details['num_questions_in_attempt'] > 0:
                denominator_for_percentage = attempt_details['num_questions_in_attempt']
        
        if denominator_for_percentage is not None and denominator_for_percentage > 0 and score is not None:
            student_percentage = (float(score) / float(denominator_for_percentage)) * 100
            attempt_details['percentage'] = student_percentage 
            if 0 <= student_percentage <= 33: student_level_display = 'Novice'
            elif student_percentage <= 66: student_level_display = 'Intermediate'
            else: student_level_display = 'Advance'
        else: attempt_details['percentage'] = None
        attempt_details['level_display'] = student_level_display
        
        
        cursor.execute("""
            SELECT q.id as question_id, q.question_text, q.option_a, q.option_b, q.option_c, q.option_d,
                   q.correct_option, q.topic, q.question_type, q.image_filename, q.marks as question_marks,
                   sta.selected_option, sta.is_correct, sta.marks_awarded
            FROM questions q
            JOIN student_test_answers sta ON q.id = sta.question_id
            WHERE q.test_id = %s AND sta.attempt_id = %s
            ORDER BY q.id ASC
        """, (attempt_details['test_id'], attempt_id))
        questions_raw = cursor.fetchall()
        
        questions_with_answers = []
        if questions_raw:
            for q_raw_item in questions_raw:
                qa_item = dict(q_raw_item)
                
                
                if qa_item.get('image_filename'):
                    qa_item['image_filename'] = str(qa_item['image_filename']).replace('\\', '/')
                    if request: 
                         qa_item['image_full_url'] = url_for('serve_upload', filename=qa_item['image_filename'], _external=True)
                         logger.info(f"PDF Report Card - Image Full URL for Q_ID {qa_item['question_id']}: {qa_item['image_full_url']}")
                    else:
                        qa_item['image_full_url'] = None 
                        logger.warning(f"PDF Report Card - No request context for Q_ID {qa_item['question_id']} image_full_url generation.")
                else:
                    qa_item['image_full_url'] = None

                
                question_type = qa_item.get('question_type')
                options_map = {
                    'A': qa_item.get('option_a'), 'B': qa_item.get('option_b'),
                    'C': qa_item.get('option_c'), 'D': qa_item.get('option_d')
                }
                if question_type == 'mcq':
                    qa_item['selected_option_text'] = options_map.get(str(qa_item.get('selected_option')).strip().upper(), qa_item.get('selected_option'))
                    qa_item['correct_option_text'] = options_map.get(str(qa_item.get('correct_option')).strip().upper())
                else: 
                    qa_item['selected_option_text'] = qa_item.get('selected_option')
                    qa_item['correct_option_text'] = qa_item.get('correct_option')
                
                questions_with_answers.append(qa_item)

        logger.info(f"Preparing PDF for report card (attempt ID {attempt_id}). Number of questions: {len(questions_with_answers)}")

        
        html_for_pdf = render_template(
            'report_card_pdf.html', 
            attempt=attempt_details, 
            questions_with_answers=questions_with_answers,
            datetime=datetime
        )
        
        pdf_bytes = HTML(string=html_for_pdf, base_url=request.url_root).write_pdf()

        safe_student_name = "".join(c if c.isalnum() else "_" for c in attempt_details.get("student_name", "student"))
        safe_test_title = "".join(c if c.isalnum() else "_" for c in attempt_details.get("test_title", "test"))
        pdf_filename = f'ReportCard_{safe_student_name}_{safe_test_title}_Attempt{attempt_id}.pdf'
        
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
        
        logger.info(f"Successfully generated PDF report card for attempt ID: {attempt_id}, Filename: {pdf_filename}")
        return response

    except Exception as e: 
        logger.error(f"Error generating PDF report card for attempt {attempt_id}: {e}", exc_info=True)
        flash(f"Could not generate PDF report card: An unexpected error occurred.", "error")
        
        return redirect(url_for('view_test_attempt_results', attempt_id=attempt_id))
    finally:
        if conn:
            if 'cursor' in locals() and cursor: cursor.close()
            conn.close()


@app.route('/download_test_review/<int:test_id>')
def download_test_review(test_id):
    if 'user_id' not in session or session.get('role') != 'student':
        flash('Unauthorized access. Please login as a student.', 'error')
        return redirect(url_for('login'))

    google_api_key = os.environ.get("GOOGLE_API_KEY", "YOUR_FALLBACK_GOOGLE_API_KEY_HERE")

    conn = None
    try:
        from weasyprint import HTML
    except ImportError as e_weasy:
        logger.error(f"WeasyPrint import error during PDF generation for test review {test_id}: {e_weasy}")
        flash("PDF generation service is unavailable due to a missing library.", "error")
        return redirect(url_for('view_test_review_with_explanations', test_id=test_id))
    except OSError as e_os_weasy: 
        logger.error(f"WeasyPrint OSError (missing C libraries) for test review {test_id}: {e_os_weasy}")
        flash("PDF generation failed: Essential system libraries for PDF creation are missing.", "error")
        return redirect(url_for('view_test_review_with_explanations', test_id=test_id))

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, title, subject, semester FROM tests WHERE id = %s", (test_id,))
        test_details = cursor.fetchone()

        if not test_details:
            flash('Test not found.', 'error')
            return redirect(url_for('dashboard'))

        cursor.execute("""
            SELECT id, question_text, option_a, option_b, option_c, option_d, 
                   correct_option, topic, question_type, image_filename, marks
            FROM questions
            WHERE test_id = %s
            ORDER BY id ASC
        """, (test_id,))
        questions_raw = cursor.fetchall()

        questions_for_pdf = []
        if questions_raw:
            for q_raw in questions_raw:
                q_detail = dict(q_raw)
                
                if q_detail.get('image_filename'):
                    q_detail['image_filename'] = str(q_detail['image_filename']).replace('\\', '/')
                    if request:
                         q_detail['image_full_url'] = url_for('serve_upload', filename=q_detail['image_filename'], _external=True)
                    else:
                        q_detail['image_full_url'] = None 
                else:
                    q_detail['image_full_url'] = None

                q_type = q_detail.get('question_type', '').lower()
                options_map = {
                    'A': q_detail.get('option_a'), 'B': q_detail.get('option_b'),
                    'C': q_detail.get('option_c'), 'D': q_detail.get('option_d')
                }
                if q_type == 'mcq':
                    q_detail['correct_answer_display'] = options_map.get(q_detail.get('correct_option'), "N/A")
                else: 
                    q_detail['correct_answer_display'] = q_detail.get('correct_option', "N/A")

                
                q_detail['ai_explanation_html'] = generate_general_ai_explanation_for_question(
                    q_detail, 
                    test_details['subject'], 
                    google_api_key 
                )
                
                questions_for_pdf.append(q_detail)
        
        logger.info(f"Preparing PDF for test review ID: {test_id}. Number of questions: {len(questions_for_pdf)}")

        html_for_pdf = render_template(
            'test_review_pdf.html', 
            test=test_details, 
            questions_data=questions_for_pdf,
            datetime=datetime 
        )
        
        pdf_bytes = HTML(string=html_for_pdf, base_url=request.url_root).write_pdf()

        safe_title = "".join(c if c.isalnum() else "_" for c in test_details["title"])
        pdf_filename = f'test_review_{safe_title}_{test_id}.pdf'
        
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{pdf_filename}"'
        
        logger.info(f"Successfully generated PDF for test review ID: {test_id}, Filename: {pdf_filename}")
        return response

    except Exception as e: 
        logger.error(f"Error generating PDF for test review ID {test_id}: {e}", exc_info=True)
        flash(f"Could not generate PDF for test review: An unexpected error occurred.", "error")
        return redirect(url_for('view_test_review_with_explanations', test_id=test_id))
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)