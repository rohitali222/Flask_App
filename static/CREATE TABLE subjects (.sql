CREATE TABLE subjects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    semester INT NOT NULL,
    department VARCHAR(50) NOT NULL,
    subject_name VARCHAR(255) NOT NULL
);

ALTER TABLE users
MODIFY id INT NOT NULL AUTO_INCREMENT;

ALTER TABLE users
ADD COLUMN semesters_taught VARCHAR(50);


DESCRIBE users;


ALTER TABLE users MODIFY subjects_per_semester TEXT;


ALTER TABLE users ADD COLUMN departments_taught VARCHAR(255);

ALTER TABLE users 
ADD COLUMN IF NOT EXISTS departments_taught VARCHAR(255) DEFAULT NULL;


UPDATE users 
SET departments_taught = department 
WHERE role = 'teacher' AND departments_taught IS NULL;


CREATE TABLE assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    semester INT NOT NULL,
    subject VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    due_date DATETIME,
    uploaded_by VARCHAR(100)
);

DROP TABLE assignments;

CREATE TABLE assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    semester VARCHAR(20) NOT NULL,
    subject VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    filename VARCHAR(255) NOT NULL,
    uploaded_by VARCHAR(100) NOT NULL,
    upload_date DATETIME NOT NULL,
    due_date DATETIME,
    FOREIGN KEY (uploaded_by) REFERENCES users(username)
);

CREATE TABLE materials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    semester VARCHAR(20) NOT NULL,
    subject VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    filename VARCHAR(255) NOT NULL,
    teacher VARCHAR(100) NOT NULL,
    upload_date DATETIME NOT NULL,
    type ENUM('notes', 'video', 'assignment', 'slides', 'other') NOT NULL,
    FOREIGN KEY (teacher) REFERENCES users(username)
);

DROP TABLE materials;

CREATE TABLE materials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    teacher VARCHAR(255),
    subject VARCHAR(255),
    type VARCHAR(100),
    title VARCHAR(255),
    description TEXT,
    filename VARCHAR(255),
    semester VARCHAR(20),
    upload_date DATETIME
);

CREATE TABLE test_attempts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    test_id INT NOT NULL,
    student_id INT NOT NULL,
    score INT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    answers TEXT NULL, -- Could store JSON of answers
    FOREIGN KEY (test_id) REFERENCES tests(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
);



ALTER TABLE materials ADD COLUMN topic VARCHAR(255) NULL;


SELECT username, department, semester, subjects FROM users WHERE username = 'student_username';

SELECT semesters_taught, subjects_per_semester FROM users WHERE username = 'tea111';

SELECT semesters_taught FROM users WHERE username = 'tea111';

SELECT subjects_per_semester FROM users WHERE username = 'tea111';

ALTER TABLE tests
ADD COLUMN teacher_id INT,
ADD CONSTRAINT fk_teacher_test FOREIGN KEY (teacher_id) REFERENCES users(id);

-- If it's missing NOT NULL and DEFAULT
ALTER TABLE users MODIFY COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending';

-- If it just needs the DEFAULT updated (and already allows NULL or is NOT NULL)
ALTER TABLE users ALTER COLUMN status SET DEFAULT 'pending';


SELECT * 
FROM tests 
WHERE teacher_username = 'u' AND semester = '8';

ALTER TABLE questions ADD COLUMN image_filename VARCHAR(255) DEFAULT NULL;

-- For the materials table
ALTER TABLE materials ADD COLUMN difficulty_level VARCHAR(20) DEFAULT NULL;

-- For the assignments table
ALTER TABLE assignments ADD COLUMN difficulty_level VARCHAR(20) DEFAULT NULL;

-- For the tests table
ALTER TABLE tests ADD COLUMN difficulty_level VARCHAR(20) DEFAULT NULL;

SET FOREIGN_KEY_CHECKS = 0;

-- Truncate or delete all tables
truncate

-- Re-enable constraints
SET FOREIGN_KEY_CHECKS = 1;

SET FOREIGN_KEY_CHECKS = 0;
DELETE FROM users;


SELECT TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME 
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
WHERE REFERENCED_TABLE_NAME = 'users';

SHOW CREATE TABLE tests;
SHOW CREATE TABLE submissions;


ALTER TABLE questions
ADD COLUMN marks DECIMAL(5,2) NULL DEFAULT 1.00 COMMENT 'Marks for this question';

ALTER TABLE test_attempts
ADD COLUMN num_questions_in_attempt INT NULL DEFAULT NULL AFTER total_marks_possible;

ALTER TABLE tests
ADD COLUMN difficulty_level VARCHAR(20) NULL DEFAULT NULL AFTER test_type;

ALTER TABLE materials
DROP COLUMN difficulty_level;


ALTER TABLE tests
ADD COLUMN difficulty_level VARCHAR(20) NULL DEFAULT NULL COMMENT 'e.g., Novice, Intermediate, Advance, General';

ALTER TABLE assignments
ADD COLUMN difficulty_level VARCHAR(20) NULL DEFAULT NULL COMMENT 'e.g., Novice, Intermediate, Advance, General';


ALTER TABLE materials
ADD COLUMN difficulty_level VARCHAR(20) NULL DEFAULT NULL COMMENT 'e.g., Novice, Intermediate, Advance, General';
