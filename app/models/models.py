
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class m_employee(db.Model):
    emp_id=db.Column(db.Integer, primary_key=True,autoincrement=True)
    first_name=db.Column(db.String(20))
    last_name= db.Column(db.String(20))
    email = db.Column(db.String(30),unique=True)
    phone_no= db.Column(db.String(10))
    password=db.Column(db.String(255))
    dob=db.Column(db.Date)
    address=db.Column(db.String(100))
    gender=db.Column(db.String(10))
    joining_date=db.Column(db.Date)
    leaving_date=db.Column(db.Date)
    department=db.Column(db.String(20))
    blood_group=db.Column(db.String(5))
    is_admin=db.Column(db.Boolean,default=False)
    first_login=db.Column(db.Boolean,default=True)
    photo=db.Column(db.String(255),default='/no_image.jpg')
    created_at=db.Column(db.DateTime,default=datetime.now)
    updated_at=db.Column(db.DateTime,onupdate=datetime.now,nullable=True)
    deleted_at=db.Column(db.DateTime,nullable=True)

    
    def is_authenticated(self):
        return True

    def is_active(self): 
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return self.emp_id

   



 

