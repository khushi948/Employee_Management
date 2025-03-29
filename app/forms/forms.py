from flask_wtf import FlaskForm
from wtforms import BooleanField, StringField,FileField,PasswordField,SubmitField
from wtforms.validators import DataRequired,Email,EqualTo

class UpdateForm(FlaskForm):

    phone_no = StringField('Phone Number', validators=[DataRequired()])
    dob = StringField('dob', validators=[DataRequired()])
    gender = StringField('gender', validators=[DataRequired()])
    department = StringField('department', validators=[DataRequired()])
    address = StringField('Address', validators=[DataRequired()])
    blood_group = StringField('blood_group', validators=[DataRequired()])
    is_admin=BooleanField('is_admin')
    photo=FileField('photo')
    submit = SubmitField('Update')

class AddForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    department = StringField('Department', validators=[DataRequired()])
    joining_date = StringField('Joining Date', validators=[DataRequired()])
    is_admin=BooleanField('is_admin')
    submit = SubmitField('Add Employee')

class LoginForm(FlaskForm):
    email=StringField('email')
    submit=SubmitField('Proceed')

class FirstLoginForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    retype_password = PasswordField(
        'Retype Password', 
        validators=[DataRequired()]
    )
    submit = SubmitField('Set Password')

class PasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Submit')
