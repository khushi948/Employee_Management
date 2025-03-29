

from flask import Flask
from flask_login import LoginManager
from flask_wtf import CSRFProtect
from app.models.models import db  
from flask_bcrypt import Bcrypt 
UPLOAD_FOLDER = '\static\images_upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SECRET_KEY='my_secret_key'
BCRYPT_LOG_ROUNDS=12
import os 




bcrypt = Bcrypt()
login_manager = LoginManager()

login_manager.login_view = 'login'
def create_app():
    app = Flask(__name__,static_folder="/app/api/static")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost/employee_management'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY']=SECRET_KEY
    app.config['BCRYPT_LOG_ROUNDS'] = BCRYPT_LOG_ROUNDS
        
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    login_manager.init_app(app)
    db.init_app(app)
    bcrypt.init_app(app)
    
    return app
     
def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

