from flask import Flask
from app import create_app
from app.models.models import db
from app.api.employee import add_employee_bp, get_details_bp,delete_emp_bp, uploaded_file_bp,view_employees_bp,edit_employees_bp,update_employees_bp,login_bp,home_bp,first_login_bp,password_bp,logout_bp

app = create_app()
app.register_blueprint(add_employee_bp, url_prefix='/api', name='add_employee')
app.register_blueprint(get_details_bp, url_prefix='/api_get', name='get_details')
app.register_blueprint(delete_emp_bp, url_prefix='/api_delete', name='delete_employee')
app.register_blueprint(view_employees_bp, url_prefix='/api_view', name='view_employees')
app.register_blueprint(update_employees_bp, url_prefix='/api_update', name='update_employees')
app.register_blueprint(edit_employees_bp, url_prefix='/api_edit', name='edit_employees')
app.register_blueprint(login_bp, url_prefix='/api_login', name='login')
app.register_blueprint(home_bp, url_prefix='/', name='home')
app.register_blueprint(first_login_bp, url_prefix='/api_first', name='first_login')
app.register_blueprint(password_bp, url_prefix='/api_password', name='password')
app.register_blueprint(logout_bp, url_prefix='/api_logout', name='logout')
app.register_blueprint(uploaded_file_bp, url_prefix='/api_upload', name='uploaded_file')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()


    app.run(debug=True)





