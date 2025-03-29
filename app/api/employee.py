import os
from datetime import datetime, timezone
from flask import Flask, abort, app,Blueprint, flash, get_flashed_messages, send_from_directory, session, url_for
from flask import redirect,request,render_template,jsonify
from flask_wtf import CSRFProtect
from app import UPLOAD_FOLDER,bcrypt,ALLOWED_EXTENSIONS
from app.models.models import m_employee,db
from app.forms.forms import FirstLoginForm, LoginForm, PasswordForm, UpdateForm,AddForm
from flask_bcrypt import Bcrypt 
from flask_login import LoginManager, login_required
from flask_login import login_user, logout_user
from flask_login import current_user,UserMixin
from werkzeug.utils import secure_filename
from app import login_manager



add_employee_bp = Blueprint('api', __name__)
get_details_bp=Blueprint('api_get',__name__)
delete_emp_bp=Blueprint('api_delete',__name__)
view_employees_bp=Blueprint('api_view',__name__)
update_employees_bp=Blueprint('api_update',__name__)
edit_employees_bp=Blueprint('api_edit',__name__)
login_bp=Blueprint('api_login',__name__)
home_bp=Blueprint('/',__name__)
first_login_bp=Blueprint('/api_first',__name__)
password_bp=Blueprint('/api_password',__name__)
logout_bp=Blueprint('/api_logout',__name__)
uploaded_file_bp=Blueprint('/api_upload',__name__)

upload_folder = UPLOAD_FOLDER
allowed_extensions = ALLOWED_EXTENSIONS

@add_employee_bp.route('/add_user', methods=['POST','GET'])
@login_required
def add_user():

    form = AddForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        first_name = form.first_name.data
        last_name = form.last_name.data
        email = form.email.data
        department = form.department.data
        joining_date = form.joining_date.data
        is_admin=form.is_admin.data

        existing_employee = m_employee.query.filter_by(email=email).first()
        if existing_employee:
            flash("Email already exists!", "danger")
            return redirect(url_for('api.add_user'))
        

        if "@" not in email or "." not in email:
            flash("Invalid email format!", "danger")
            return redirect(url_for('api.add_user'))
        
        emp = m_employee(
            first_name=first_name,
            last_name=last_name,
            email=email,
            department=department,
            joining_date=joining_date,
            created_at=datetime.now(timezone.utc),
            is_admin=is_admin
        )
        db.session.add(emp)

        db.session.commit()

        flash("Employee added successfully!", "success")
        session['redirect_after_flash'] = url_for('api_view.view_employees')

        return render_template('add_employee.html', form=form)  

    return render_template('add_employee.html', form=form)


@get_details_bp.route('/get_details/<int:emp_id>',methods=['GET'])
@login_required
def get_details(emp_id):
    detail= m_employee.query.get(emp_id)
    response={}
    if detail.deleted_at==None:
        response["message"]="employee details retrieved successfully"
        response['status'] = 200
        response['data']={'emp_id':detail.emp_id, 'first_name':detail.first_name,'last_name':detail.last_name,'email':detail.email,'phone_no':detail.phone_no,'dob':detail.dob,'address':detail.address,'department':detail.department,'gender':detail.gender,'joining_date':detail.joining_date,'photo':detail.photo,'dob':detail.dob,'joining_date':detail.joining_date,'blood_group':detail.blood_group}
        emp={'emp_id':detail.emp_id, 'first_name':detail.first_name,'last_name':detail.last_name,'email':detail.email,'phone_no':detail.phone_no,'dob':detail.dob,'address':detail.address,'department':detail.department,'gender':detail.gender,'joining_date':detail.joining_date,'photo':detail.photo,'dob':detail.dob,'joining_date':detail.joining_date,'blood_group':detail.blood_group}
    else:
        response['message']=['employee not found']
        response['status']=404
   
    return render_template('employee_profile.html', emp=detail)



@delete_emp_bp.route('/delete_emp/<int:emp_id>',methods=['GET'])
@login_required
def delete_emp(emp_id):
    emp=m_employee.query.get_or_404(emp_id)
    emp = m_employee.query.get_or_404(emp_id)

    if emp.deleted_at is None:
        emp.deleted_at = datetime.now(timezone.utc)
        db.session.commit()
        flash("Employee deleted successfully.", "success")
    else:
        flash("Employee already deleted.", "error")
    
    return redirect(url_for('api_view.view_employees'))


@view_employees_bp.route('/view_employees',methods=['GET'])
@login_required
def view_employees():
    emp=m_employee.query.filter_by(deleted_at=None).all()
    emp_list=[]
    for e in emp:
        emp_list.append({'emp_id':e.emp_id,
                         'first_name':e.first_name,
                         'last_name':e.last_name,
                         'email':e.email,
                         'phone_no':e.phone_no,
                         'department':e.department,
                         'photo':e.photo
                         })
    
    if current_user.is_admin:                
        return render_template('employee_list_admin.html',emp_list=emp_list,emp=emp)
    else:
        return render_template('employee_list.html',emp_list=emp_list,emp=emp)    

@edit_employees_bp.route('/edit_employees/<int:emp_id>', methods=['GET'])
@login_required
def edit_employees(emp_id):
    emp = m_employee.query.get(emp_id)

    if not emp:
        flash("Employee not found!", "danger")
        return redirect(url_for('api_view.view_employees')) 
    form = UpdateForm(obj=emp)  
    return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)


@update_employees_bp.route('/update_employees/<int:emp_id>', methods=['POST'])
@login_required
def update_employees(emp_id):

    emp = m_employee.query.get(emp_id)
    if not emp:
        flash("Employee not found!", "error")
        return redirect(url_for('api_view.view_employees'))  

    form = UpdateForm(request.form)
    if form.phone_no.data and len(form.phone_no.data) != 10:
        flash("Phone number must be exactly 10 digits.", "error")
        return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)
    if form.dob.data:
        try:
            birth_date = datetime.strptime(form.dob.data, "%Y-%m-%d")
        except ValueError:
            flash("Invalid date format. Please use YYYY-MM-DD.", "error")
            return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)

        today = datetime.now()
        age = today.year - birth_date.year
       
        if today.month < birth_date.month or (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1

        if age < 18:
            flash("Employee must be at least 18 years old.", "error")
            return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)
    if form.validate_on_submit():
        try:
            emp.phone_no = form.phone_no.data
            emp.address = form.address.data
            emp.gender = form.gender.data
            emp.blood_group = form.blood_group.data
            emp.department = form.department.data
            emp.dob = form.dob.data
            emp.updated_at = datetime.now(timezone.utc)
            emp.is_admin = form.is_admin.data
            db.session.commit()

            flash("Employee profile updated successfully!", "success")
        except Exception as e:
            flash(f"An error occurred while updating the employee: {str(e)}", "error")
            return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)

    f = request.files.get('photo')
    if f:
        fname = secure_filename(f.filename)
        if fname != '':
            try:
                BASE_DIR = os.path.abspath(os.path.dirname(__file__))  
                UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'image_uploads') 

                file_path = os.path.join(UPLOAD_FOLDER, fname)
                emp.photo = fname
                f.save(file_path)

                flash("Profile photo uploaded successfully!", "success")
            except Exception as e:
                flash(f"Error saving the photo: {str(e)}", "danger")
                return render_template('edit_employee.html', form=form, emp_id=emp_id, emp=emp)
        db.session.commit()

    else:
        flash("No new photo uploaded.", "info")

    return redirect(url_for('api_view.view_employees'))  

@home_bp.route('/')
def home():
    return render_template('home.html')

@login_bp.route('/login', methods=['POST','GET'])
def login():
    form = LoginForm()
    if request.method == "POST" and form.validate_on_submit():
        email = form.email.data
        if "@" not in email or "." not in email:
            flash("Invalid email format!", "danger")
            return render_template('login.html', form=form)
        emp = m_employee.query.filter_by(email=email).first()

        if not emp:
            flash("Email does not exist!", "danger")
            return render_template('login.html', form=form)
        flash("Correct Email!", "success")

        if emp.first_login:
            flash("Welcome! This is your first login.", "info")
            emp.first_login = False
            emp.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            session['redirect_after_flash'] = url_for('/api_first.first_login', emp_id=emp.emp_id)

        else:
            session['redirect_after_flash'] = url_for('/api_password.password', emp_id=emp.emp_id)

        return render_template('login.html', form=form)  

    return render_template('login.html', form=form)
	
@first_login_bp.route('/first_login/<int:emp_id>',methods=['POST','GET'])
def first_login(emp_id):
    emp = m_employee.query.filter_by(emp_id=emp_id).first()
    form = FirstLoginForm()
    
    if request.method == 'POST' and form.validate_on_submit():
        password = form.password.data
        retype_password = form.retype_password.data
        if len(password) < 8:
            flash("Password must be at least 8 characters long!", "danger")
            return redirect(url_for('/api_first.first_login', emp_id=emp_id))
        if password != retype_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('/api_first.first_login', emp_id=emp_id))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        emp.password = hashed_password
        emp.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        flash("Password set successfully! Please log in.", "success")
        session['redirect_after_flash'] = url_for('api_login.login') 

        return render_template('first_login.html', form=form)  
    return render_template('first_login.html', form=form)


@login_manager.user_loader
def load_user(emp_id):

    return m_employee.query.get(emp_id)

@password_bp.route('/password/<int:emp_id>',methods=['GET','POST'])
def password(emp_id):
    emp = m_employee.query.filter_by(emp_id=emp_id).first()
    if not emp:
        flash("Employee not found!", "danger")
        return redirect(url_for('api_login.login'))

    form = PasswordForm()

    if request.method == 'POST' and form.validate_on_submit():
        password = form.password.data
        if bcrypt.check_password_hash(emp.password, password):
            login_user(emp, remember=True)
            flash("Login successful!", "success")
            session['redirect_after_flash'] = url_for('api_view.view_employees')
            return render_template('password.html', form=form, user=current_user)  
        
        if len(password) < 8:
            flash("Password must be at least 8 characters long!", "danger")
            return redirect(url_for('/api_password.password', emp_id=emp_id))
        else:
            flash("Wrong password!", "danger")
            return redirect(url_for('/api_password.password', emp_id=emp_id))
        
    return render_template('password.html', form=form, user=current_user)
@logout_bp.route('/logout',methods=['GET','POST'])
@login_required
def logout():
    session.clear()
    logout_user()
    
    return redirect(url_for("/.home"))

@uploaded_file_bp.route('/uploaded_file/<filename>',methods=['GET','POST'])
@login_required
def uploaded_file(filename):
    return send_from_directory("api/static/image_uploads", filename)
 