from flask import Flask
from flask_login import LoginManager
from flask_wtf import CSRFProtect
from app.models.models import db  
from flask_bcrypt import Bcrypt 

UPLOAD_FOLDER = r'static/images_upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SECRET_KEY='my_secret_key'
BCRYPT_LOG_ROUNDS=12
import os 

bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'api_login.login'

def create_app(test_config=None):
    app = Flask(__name__,static_folder="/app/api/static")
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost/employee_management'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY']=SECRET_KEY
    app.config['BCRYPT_LOG_ROUNDS'] = BCRYPT_LOG_ROUNDS
    
    
    if test_config:
        app.config.update(test_config)

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    login_manager.init_app(app)
    db.init_app(app)
    bcrypt.init_app(app)
    
    # Import blueprints here to avoid circular imports
    from app.api.employee import (
        add_employee_bp, get_details_bp, delete_emp_bp, view_employees_bp,
        update_employees_bp, edit_employees_bp, login_bp, home_bp,
        first_login_bp, password_bp, logout_bp, uploaded_file_bp
    )
    
    # Register blueprints
    app.register_blueprint(add_employee_bp)
    app.register_blueprint(get_details_bp)
    app.register_blueprint(delete_emp_bp)
    app.register_blueprint(view_employees_bp)
    app.register_blueprint(update_employees_bp)
    app.register_blueprint(edit_employees_bp)
    app.register_blueprint(login_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(first_login_bp)
    app.register_blueprint(password_bp)
    app.register_blueprint(logout_bp)
    app.register_blueprint(uploaded_file_bp)
    
    return app
    
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

