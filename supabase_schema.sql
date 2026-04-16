-- =============================================================
-- Deepfake Detection System — Supabase PostgreSQL Schema
-- Run this in your Supabase project > SQL Editor
-- =============================================================

-- ----------------------------------------------------------------
-- USERS
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    password_hash TEXT,                        -- NULL for Google OAuth users
    role          TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')),
    google_id     TEXT UNIQUE,
    avatar_url    TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

-- Admin default (remember to update the hash via the app seed)
INSERT INTO users (email, name, password_hash, role)
VALUES ('admin@deepfake.com', 'Administrator', 'SEED_VIA_APP', 'admin')
ON CONFLICT (email) DO NOTHING;

-- ----------------------------------------------------------------
-- RECORDINGS (15-sec baseline)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recordings (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_path   TEXT NOT NULL,
    audio_path   TEXT NOT NULL,
    duration_sec FLOAT NOT NULL DEFAULT 15,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- EXAM SESSIONS
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exam_sessions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recording_id   UUID REFERENCES recordings(id),
    start_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time       TIMESTAMPTZ,
    status         TEXT NOT NULL DEFAULT 'active'
                       CHECK (status IN ('active','completed','suspended','flagged')),
    exam_type      TEXT NOT NULL DEFAULT 'exam',
    score          INT,
    total_questions INT
);

-- ----------------------------------------------------------------
-- DEEPFAKE RESULTS (per-frame analysis stored as aggregate per session)
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS deepfake_results (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id     UUID NOT NULL REFERENCES exam_sessions(id) ON DELETE CASCADE,
    user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    video_score    FLOAT NOT NULL DEFAULT 0,    -- 0=real, 1=deepfake
    audio_score    FLOAT NOT NULL DEFAULT 0,
    fusion_score   FLOAT NOT NULL DEFAULT 0,
    similarity_score FLOAT NOT NULL DEFAULT 1,  -- 1=identical, 0=different
    is_deepfake    BOOLEAN NOT NULL DEFAULT FALSE,
    is_suspicious  BOOLEAN NOT NULL DEFAULT FALSE,
    confidence     FLOAT NOT NULL DEFAULT 0,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- ACTIVITY LOGS
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activity_logs (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES exam_sessions(id) ON DELETE CASCADE,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,   -- e.g. 'tab_switch','blur','fullscreen_exit','deepfake_alert'
    details    JSONB,
    timestamp  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------
-- QUESTIONS
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS questions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    exam_type      TEXT NOT NULL DEFAULT 'exam',
    question_text  TEXT NOT NULL,
    options        JSONB NOT NULL DEFAULT '[]',   -- ["A","B","C","D"]
    correct_answer TEXT NOT NULL,
    marks          INT NOT NULL DEFAULT 1,
    difficulty     TEXT DEFAULT 'medium' CHECK (difficulty IN ('easy','medium','hard'))
);

-- 30 exam questions (5 original + 25 new)
INSERT INTO questions (exam_type, question_text, options, correct_answer, marks) VALUES
-- Q1-Q5: Original
('exam', 'What does CNN stand for in deep learning?',
 '["Convolutional Neural Network","Central Neural Node","Cognitive Network Node","Computed Neural Net"]',
 'Convolutional Neural Network', 2),
('exam', 'Which library is used for audio feature extraction in this system?',
 '["TensorFlow","OpenCV","Librosa","PyTorch"]',
 'Librosa', 1),
('exam', 'What is MFCC?',
 '["Mel Frequency Cepstral Coefficients","Multi-Feature Computer Coding","Machine Feature Comparison Chart","Multiple Fusion CNN Classifier"]',
 'Mel Frequency Cepstral Coefficients', 2),
('exam', 'Which fusion method is used for combining audio and video scores?',
 '["Feature-level fusion","Decision-level fusion","Score-level fusion","Model-level fusion"]',
 'Score-level fusion', 2),
('exam', 'What does LSTM stand for?',
 '["Long Short-Term Memory","Large Scale Training Model","Linear Sigmoid Transfer Method","Layered Signal Transfer Mode"]',
 'Long Short-Term Memory', 1),

-- Q6-Q10: Deep Learning Fundamentals
('exam', 'Which activation function is most commonly used in the hidden layers of a deep neural network?',
 '["Sigmoid","Tanh","ReLU","Softmax"]',
 'ReLU', 1),
('exam', 'What technique is used to prevent overfitting by randomly disabling neurons during training?',
 '["Batch Normalization","Dropout","Weight Decay","Data Augmentation"]',
 'Dropout', 1),
('exam', 'In a GAN (Generative Adversarial Network), which component attempts to distinguish real data from fake?',
 '["Generator","Discriminator","Encoder","Decoder"]',
 'Discriminator', 2),
('exam', 'What does the term "transfer learning" refer to?',
 '["Training a model from scratch on a new dataset","Reusing a pre-trained model on a new but related task","Moving model weights between GPUs","Converting a model from one framework to another"]',
 'Reusing a pre-trained model on a new but related task', 2),
('exam', 'Which loss function is typically used for binary classification tasks?',
 '["Mean Squared Error","Categorical Cross-Entropy","Binary Cross-Entropy","Hinge Loss"]',
 'Binary Cross-Entropy', 1),

-- Q11-Q15: Computer Vision
('exam', 'What does the term "deepfake" refer to?',
 '["A secure hashing algorithm","AI-generated synthetic media replacing a real person","A type of network intrusion","A compression artefact in video"]',
 'AI-generated synthetic media replacing a real person', 1),
('exam', 'Which OpenCV function is used to detect faces using a Haar Cascade classifier?',
 '["cv2.findContours","cv2.detectMultiScale","cv2.threshold","cv2.matchTemplate"]',
 'cv2.detectMultiScale', 2),
('exam', 'What does FFT stand for in signal processing?',
 '["Fast Feature Transform","Forward Fourier Technique","Fast Fourier Transform","Frequency Feature Tensor"]',
 'Fast Fourier Transform', 1),
('exam', 'In image processing, what does converting an image to grayscale achieve?',
 '["Increases resolution","Reduces colour channels from 3 to 1, simplifying computation","Adds depth information","Applies Gaussian blur"]',
 'Reduces colour channels from 3 to 1, simplifying computation', 1),
('exam', 'Which technique is used to normalise pixel intensity distributions in images?',
 '["Edge Detection","Histogram Equalisation","Dilation","Erosion"]',
 'Histogram Equalisation', 2),

-- Q16-Q20: Python & Libraries
('exam', 'Which Python library provides the ndarray object used for numerical computation?',
 '["pandas","matplotlib","NumPy","SciPy"]',
 'NumPy', 1),
('exam', 'What does the Flask decorator @app.route("/") define?',
 '["A database model","A middleware function","A URL endpoint and its handler","A static file path"]',
 'A URL endpoint and its handler', 1),
('exam', 'Which function in OpenCV reads an image file from disk?',
 '["cv2.imshow","cv2.imwrite","cv2.imread","cv2.VideoCapture"]',
 'cv2.imread', 1),
('exam', 'In Python, what does the keyword "yield" do inside a function?',
 '["Returns a value and terminates the function","Pauses the function and returns a value, creating a generator","Raises a StopIteration exception","Imports a module"]',
 'Pauses the function and returns a value, creating a generator', 2),
('exam', 'Which Python library is used to interact with a Supabase/PostgreSQL database using service keys?',
 '["psycopg2","supabase-py","SQLAlchemy","pymongo"]',
 'supabase-py', 1),

-- Q21-Q25: AI Security & Proctoring
('exam', 'What does "liveness detection" aim to verify in a biometric system?',
 '["Network speed","That the biometric sample comes from a live person, not a spoof","The frame rate of a camera","Audio clarity"]',
 'That the biometric sample comes from a live person, not a spoof', 2),
('exam', 'Which signal can be used to non-contact measure a person''s heart rate from a webcam video?',
 '["GLCM","FFT","rPPG (remote Photoplethysmography)","MFCC"]',
 'rPPG (remote Photoplethysmography)', 2),
('exam', 'In online exam proctoring, what is the purpose of a "gaze detection" system?',
 '["To improve video quality","To detect if the student looks away from the screen","To compress video data","To verify internet speed"]',
 'To detect if the student looks away from the screen', 1),
('exam', 'What is the role of cosine similarity in face verification?',
 '["Measures pixel brightness","Computes the angle between two face embedding vectors to assess similarity","Detects edges in a face image","Removes background noise"]',
 'Computes the angle between two face embedding vectors to assess similarity', 2),
('exam', 'Which attack type does a deepfake detection system primarily guard against?',
 '["SQL Injection","Man-in-the-Middle attack","Presentation attack using AI-generated video","Brute-force password attack"]',
 'Presentation attack using AI-generated video', 2),

-- Q26-Q30: Advanced AI/ML
('exam', 'What does "batch normalization" do in a neural network?',
 '["Shuffles training data","Normalises the inputs of each layer to stabilise and speed up training","Reduces the learning rate","Removes duplicate training samples"]',
 'Normalises the inputs of each layer to stabilise and speed up training', 2),
('exam', 'Which metric measures the proportion of actual positives correctly identified by a classifier?',
 '["Precision","F1-Score","Recall","Specificity"]',
 'Recall', 2),
('exam', 'What is the purpose of an autoencoder in unsupervised learning?',
 '["To classify images","To learn a compressed representation (encoding) of input data","To generate labelled datasets","To perform object detection"]',
 'To learn a compressed representation (encoding) of input data', 2),
('exam', 'In the context of deepfake detection, what does the term "temporal consistency" refer to?',
 '["Uniform file timestamps","Smoothness and continuity of features (e.g. skin texture) across video frames","The frame rate of the recording","Audio synchronisation"]',
 'Smoothness and continuity of features (e.g. skin texture) across video frames', 2),
('exam', 'Which regularisation technique adds a penalty proportional to the absolute value of model weights?',
 '["L2 Regularisation (Ridge)","Dropout","L1 Regularisation (Lasso)","Batch Normalization"]',
 'L1 Regularisation (Lasso)', 2);

-- ----------------------------------------------------------------
-- INDEXES
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_recordings_user_id      ON recordings(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id        ON exam_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status         ON exam_sessions(status);
CREATE INDEX IF NOT EXISTS idx_deepfake_results_session ON deepfake_results(session_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_session   ON activity_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_user      ON activity_logs(user_id);

-- ----------------------------------------------------------------
-- Row-Level Security (Supabase)
-- ----------------------------------------------------------------
ALTER TABLE users           ENABLE ROW LEVEL SECURITY;
ALTER TABLE recordings      ENABLE ROW LEVEL SECURITY;
ALTER TABLE exam_sessions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE deepfake_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs   ENABLE ROW LEVEL SECURITY;
ALTER TABLE questions       ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (backend uses service role key)
CREATE POLICY "Service role full access" ON users           FOR ALL USING (TRUE);
CREATE POLICY "Service role full access" ON recordings      FOR ALL USING (TRUE);
CREATE POLICY "Service role full access" ON exam_sessions   FOR ALL USING (TRUE);
CREATE POLICY "Service role full access" ON deepfake_results FOR ALL USING (TRUE);
CREATE POLICY "Service role full access" ON activity_logs   FOR ALL USING (TRUE);
CREATE POLICY "Service role full access" ON questions       FOR ALL USING (TRUE);
