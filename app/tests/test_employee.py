import pytest
from flask import url_for
from app.models.models import m_employee, db
from datetime import datetime, timezone
from werkzeug.security import check_password_hash
from flask_bcrypt import Bcrypt

@pytest.fixture
def add_test_user(app):
    """Fixture to create a test admin user"""
    with app.app_context():
        bcrypt = Bcrypt(app)
        hashed_password = bcrypt.generate_password_hash('Test@123456').decode('utf-8')
        
        admin = m_employee(
            first_name="Admin",
            last_name="User",
            email="admin@gmail.com",
            department="IT",
            joining_date=datetime.now(timezone.utc),
            is_admin=True,
            photo="default.jpg",
            password=hashed_password,
            first_login=False
        )
        
        db.session.add(admin)
        db.session.commit()
        
        admin_id = admin.emp_id
        db.session.expunge_all()
        admin = db.session.get(m_employee, admin_id)
        
        return admin

@pytest.fixture(scope='function')
def logged_in_client(client, add_test_user, app):
    """Fixture to create a logged-in client with admin user"""
    with app.app_context():
        with client.session_transaction() as sess:
            sess['_user_id'] = str(add_test_user.emp_id)
        return client

@pytest.fixture
def test_employees_login(app):
    with app.app_context():
        bcrypt = Bcrypt(app)
        hashed_password = bcrypt.generate_password_hash('Test@123456').decode('utf-8')

        emp1 = m_employee(
            first_name="John",
            last_name="Doe",
            email="john2@gmail.com",
            department="IT",
            joining_date=datetime.now(timezone.utc),
            is_admin=True,
            photo="default.jpg",
            password=hashed_password,
            first_login=False
        )
        
        emp2 = m_employee(
            first_name="Jane",
            last_name="Smith",
            email="jane2@gmail.com",
            department="HR",
            joining_date=datetime.now(timezone.utc),
            is_admin=False,
            photo="default.jpg",
            first_login=True  
        )

        emp3 = m_employee(
            first_name="John",
            last_name="smith",
            email="johnsmith@gmail.com",
            department="HR",
            joining_date=datetime.now(timezone.utc),
            is_admin=False,
            photo="default.jpg",
            first_login=True 
        )
        
        db.session.add(emp1)
        db.session.add(emp2)
        db.session.add(emp3)
        db.session.commit()
        
        return [emp1.emp_id, emp2.emp_id, emp3.emp_id]


@pytest.fixture
def test_employees(app):
    with app.app_context():
        emp1 = m_employee(
            first_name="John",
            last_name="Doe",
            email="john2@gmail.com",
            department="IT",
            joining_date=datetime.now(timezone.utc),
            is_admin=True,
            photo="default.jpg"
        )
        
        emp2 = m_employee(
            first_name="Jane",
            last_name="Smith",
            email="jane2@gmail.com",
            department="HR",
            joining_date=datetime.now(timezone.utc),
            is_admin=False,
            photo="default.jpg"
        )

        emp3=m_employee(
            first_name="John",
            last_name="smith",
            email="johnsmith@gmail.com",
            department="HR",
            joining_date=datetime.now(timezone.utc),
            is_admin=False,
            photo="default.jpg"
        )
        
        db.session.add(emp1)
        db.session.add(emp2)
        db.session.add(emp3)
        db.session.commit()
        
        return [emp1.emp_id, emp2.emp_id,emp3.emp_id]



def test_add_user_post_valid(logged_in_client, app):
    """Test adding a new employee with valid data"""
    with app.app_context():
        data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john1test2@gmail.com',
            'department': 'HR',
            'joining_date': '2023-01-01',
            'is_admin': False
        }
        response = logged_in_client.post('/add_user', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee added successfully" in response.data


def test_view_employees_as_admin(client, test_employees):
    """Test viewing employees as admin user"""
    admin_id = test_employees[0]
    with client.application.app_context():
        admin = m_employee.query.get(admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    response = client.get('/view_employees')
    assert response.status_code == 200
    assert b'Add employee' in response.data  
    assert b'John Doe' in response.data
    assert b'Jane Smith' in response.data
    assert b'IT' in response.data
    assert b'HR' in response.data

def test_view_employees_as_regular_user(client, test_employees):
    """Test viewing employees as regular user"""
    regular_user = test_employees[1]  
    with client.application.app_context():
        user = m_employee.query.get(regular_user)
        with client.session_transaction() as session:
            session['_user_id'] = user.emp_id
    
    response = client.get('/view_employees')
    assert response.status_code == 200
    assert b'Add employee' not in response.data
    assert b'John Doe' in response.data
    assert b'Jane Smith' in response.data
    assert b'IT' in response.data
    assert b'HR' in response.data

def test_view_employees_deleted_employee(client, test_employees):
    """Test that deleted employees are not shown in the list"""
    emp_to_delete_id = test_employees[1]
    emp_to_delete = db.session.get(m_employee, emp_to_delete_id)
    emp_to_delete.deleted_at = datetime.now(timezone.utc)
    db.session.commit()
    admin_id = test_employees[0] 
    with client.application.app_context():
        admin = db.session.get(m_employee, admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    
    response = client.get('/view_employees')
    assert response.status_code == 200
    assert b'John Doe' in response.data
    assert b'Jane Smith' not in response.data

def test_add_employee_as_admin(client, test_employees):
    """Test adding an employee as an admin user"""
    admin_id = test_employees[0] 
    with client.application.app_context():
        admin = db.session.get(m_employee, admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    
    data = {
        'first_name': 'New',
        'last_name': 'Employee',
        'email': 'newemployee@gmail.com',
        'department': 'IT',
        'joining_date': '2023-01-01',
        'is_admin': False
    }
    response = client.post('/add_user', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Employee added successfully" in response.data

def test_add_employee_duplicate_email(client, test_employees):
    """Test adding an employee with duplicate email"""
    admin_id = test_employees[0] 
    with client.application.app_context():
        admin = db.session.get(m_employee, admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    
    
    data = {
        'first_name': 'New',
        'last_name': 'Employee',
        'email': 'jane2@gmail.com', 
        'department': 'IT',
        'joining_date': '2023-01-01',
        'is_admin': False
    }
    response = client.post('/add_user', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Email already exists" in response.data

def test_add_employee_invalid_email_format(client, test_employees):
    """Test adding an employee with invalid email format (bypassing frontend validation)"""
    admin_id = test_employees[0] 
    with client.application.app_context():
        admin = db.session.get(m_employee, admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    data = {
        'first_name': 'New',
        'last_name': 'Employee',
        'email': 'invalid-email',  
        'department': 'IT',
        'joining_date': '2023-01-01',
        'is_admin': False
    }
    response = client.post('/add_user', data=data, follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid email format" in response.data

def test_delete_employee_as_admin(client, test_employees):
    """Test deleting an employee as admin user"""
    admin_id = test_employees[0]  
    with client.application.app_context():
        admin = db.session.get(m_employee, admin_id)
        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id
    
    response = client.get(f'/delete_emp/{test_employees[1]}')
    assert response.status_code == 302 
    with client.application.app_context():
        deleted_emp = db.session.get(m_employee, test_employees[1])
        assert deleted_emp.deleted_at is not None

def test_delete_employee_as_regular_user(client, test_employees):
    """Test that regular users cannot delete employees"""
    regular_user_id = test_employees[1] 
    with client.application.app_context():
        regular_user = db.session.get(m_employee, regular_user_id)
        with client.session_transaction() as session:
            session['_user_id'] = regular_user.emp_id
    response = client.get(f'/delete_emp/{test_employees[0]}')
    assert response.status_code == 403
    with client.application.app_context():
        employee = db.session.get(m_employee, test_employees[0])
        assert employee.deleted_at is None

def test_first_login_flow(client, test_employees):
    """Test complete first login flow"""
    
    email_response = client.post('/login', data={
        'email': 'jane2@gmail.com'
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Correct Email!" in email_response.data

    with client.application.app_context():
        emp = m_employee.query.get(test_employees[1])
        if emp.first_login:
            first_login_response = client.post(f'/first_login/{test_employees[1]}', data={
                'password': 'Test@123456',
                'retype_password': 'Test@123456'
            }, follow_redirects=True)
            assert first_login_response.status_code == 200
            assert b"Password set successfully!" in first_login_response.data

            emp = m_employee.query.get(test_employees[1])
            assert emp.first_login == False

            password_response = client.post(f'/password/{test_employees[1]}', data={
                'password': 'Test@123456'
            }, follow_redirects=True)
            assert password_response.status_code == 200
            assert b"Login successful!" in password_response.data


def test_regular_login_flow(client, test_employees_login):
    """Test regular login flow for user who already has password set"""
    email_response = client.post('/login', data={
        'email': 'john2@gmail.com' 
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Correct Email!" in email_response.data

    with client.application.app_context():
        emp = m_employee.query.get(test_employees_login[0])  
        assert emp.first_login == False
        assert emp.password is not None 

      
        password_response = client.post(f'/password/{test_employees_login[0]}', data={ 
            'password': 'Test@123456'
        }, follow_redirects=True)
        assert password_response.status_code == 200
        assert b"Login successful!" in password_response.data
def test_wrong_password_login(client, test_employees_login):
    """Test login with wrong password"""
   
    email_response = client.post('/login', data={
        'email': 'john2@gmail.com'
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Correct Email!" in email_response.data
 
    password_response = client.post(f'/password/{test_employees_login[0]}', data={
        'password': 'WrongPassword123!'
    }, follow_redirects=True)
    assert password_response.status_code == 200
    assert b"Wrong password!" in password_response.data



def test_password_validation_during_first_login(client, test_employees):
    """Test password validation during first login"""
    
    email_response = client.post('/login', data={
        'email': 'jane2@gmail.com'
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Welcome! This is your first login." in email_response.data

    short_password_response = client.post(f'/first_login/{test_employees[1]}', data={
        'password': 'short',
        'retype_password': 'short'
    }, follow_redirects=True)
    assert short_password_response.status_code == 200
    assert b"Password must be at least 8 characters long!" in short_password_response.data

    no_upper_response = client.post(f'/first_login/{test_employees[1]}', data={
        'password': 'password123',
        'retype_password': 'password123'
    }, follow_redirects=True)
    assert no_upper_response.status_code == 200
    assert b"Password must contain at least one uppercase letter." in no_upper_response.data

    no_special_response = client.post(f'/first_login/{test_employees[1]}', data={
        'password': 'Password123',
        'retype_password': 'Password123'
    }, follow_redirects=True)
    assert no_special_response.status_code == 200
    assert b"Password must contain at least one special character." in no_special_response.data

    valid_password_response = client.post(f'/first_login/{test_employees[1]}', data={
        'password': 'Test@123456',
        'retype_password': 'Test@123456'
    }, follow_redirects=True)
    assert valid_password_response.status_code == 200
    
    assert b"Employee List" in valid_password_response.data
    
    with client.application.app_context():
        bcrypt = Bcrypt(client.application)
        emp = m_employee.query.get(test_employees[1])
        assert emp.first_login == False
        assert emp.password is not None
        assert bcrypt.check_password_hash(emp.password, 'Test@123456')

    password_response = client.post(f'/password/{test_employees[1]}', data={
        'password': 'Test@123456'
    }, follow_redirects=True)
    assert password_response.status_code == 200
    assert b"Login successful!" in password_response.data

def test_wrong_password_login(client, test_employees_login):
    """Test login with wrong password"""
    email_response = client.post('/login', data={
        'email': 'john2@gmail.com'
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Correct Email!" in email_response.data

    password_response = client.post(f'/password/{test_employees_login[0]}', data={
        'password': 'WrongPassword123!'
    }, follow_redirects=True)
    assert password_response.status_code == 200
    assert b"Wrong password!" in password_response.data

def test_short_password_login(client, test_employees_login):
    """Test login with password shorter than 8 characters"""
    email_response = client.post('/login', data={
        'email': 'john2@gmail.com'
    }, follow_redirects=True)
    assert email_response.status_code == 200
    assert b"Correct Email!" in email_response.data

    password_response = client.post(f'/password/{test_employees_login[0]}', data={
        'password': 'short'
    }, follow_redirects=True)
    assert password_response.status_code == 200
    assert b"Password must be at least 8 characters long!" in password_response.data

def test_edit_employee_as_admin(client, test_employees, app):
    """Test editing an employee as admin user"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        # First get the edit form
        get_response = client.get(f'/edit_employees/{test_employees[1]}')
        assert get_response.status_code == 200

        # Extract CSRF token from the form
        csrf_token = None
        if b'name="csrf_token"' in get_response.data:
            csrf_token = get_response.data.decode('utf-8').split('name="csrf_token" value="')[1].split('"')[0]

        data = {
            'phone_no': '1234567890',
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '1990-01-01',
            'is_admin': False
        }
        if csrf_token:
            data['csrf_token'] = csrf_token

        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee profile updated successfully!" in response.data

def test_edit_employee_validation(client, test_employees, app):
    """Test employee edit form validation"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        data = {
            'phone_no': '123',  
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '1990-01-01',
            'is_admin': False
        }
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Phone number must be exactly 10 digits" in response.data

        data['phone_no'] = '1234567890'
        data['dob'] = '01-01-1990' 
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Invalid date format" in response.data

        data['dob'] = '2020-01-01'  
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee must be at least 18 years old" in response.data

def test_password_mismatch_during_first_login(client, test_employees, app):
    """Test password mismatch during first login"""
    with app.app_context():
        email_response = client.post('/login', data={
            'email': 'jane2@gmail.com'
        }, follow_redirects=True)
        assert email_response.status_code == 200
        assert b"Welcome! This is your first login." in email_response.data

        response = client.post(f'/first_login/{test_employees[1]}', data={
            'password': 'Test@123456',
            'retype_password': 'DifferentPassword123!'
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b"Passwords do not match!" in response.data

def test_invalid_password_format(client, test_employees, app):
    """Test invalid password format during first login"""
    with app.app_context():
        email_response = client.post('/login', data={
            'email': 'jane2@gmail.com'
        }, follow_redirects=True)
        assert email_response.status_code == 200
        assert b"Welcome! This is your first login." in email_response.data

        response = client.post(f'/first_login/{test_employees[1]}', data={
            'password': 'Test@Password',
            'retype_password': 'Test@Password'
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b"Password must contain at least one number" in response.data

def test_view_employee_profile(client, test_employees, app):
    """Test viewing employee profile"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        response = client.get(f'/get_details/{test_employees[1]}')
        assert response.status_code == 200
        
        emp = m_employee.query.get(test_employees[1])
        assert emp.first_name.encode() in response.data
        assert emp.last_name.encode() in response.data
        assert emp.email.encode() in response.data
        assert emp.department.encode() in response.data
        
        assert b"Employee Profile" in response.data
        assert b"Edit Profile" in response.data
        assert b"Back to list" in response.data

def test_upload_employee_photo(client, test_employees, app):
    """Test uploading employee photo"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        get_response = client.get(f'/edit_employees/{test_employees[1]}')
        assert get_response.status_code == 200
        csrf_token = get_response.data.decode('utf-8').split('name="csrf_token" value="')[0].split('"')[0]

        
        import io
        test_image = io.BytesIO(b"fake image data")
        test_image.name = "test.jpg"

        response = client.post(
            f'/update_employees/{test_employees[0]}',
            data={
                'photo': (test_image, 'test.jpg'),
                'csrf_token': csrf_token
            },
            content_type='multipart/form-data',
            follow_redirects=True
        )
        
        assert response.status_code == 200
        assert b"Profile photo uploaded successfully!" in response.data

def test_logout(client, test_employees, app):
    """Test logout functionality"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        response = client.get('/view_employees')
        assert response.status_code == 200
        assert b"Hello" in response.data

        response = client.get('/logout', follow_redirects=True)
        assert response.status_code == 200
        assert b"Login" in response.data

        response = client.get('/view_employees')
        assert response.status_code == 302  

def test_edit_employee_invalid_date_format(client, test_employees, app):
    """Test editing employee with invalid date format"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        data = {
            'phone_no': '1234567890',
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '01-01-1990',  
            'is_admin': False
        }
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Invalid date format" in response.data

def test_edit_employee_underage(client, test_employees, app):
    """Test editing employee with underage date of birth"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        data = {
            'phone_no': '1234567890',
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '2020-01-01',  
            'is_admin': False
        }
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee must be at least 18 years old" in response.data

def test_view_employees_empty_list(client, app):
    """Test viewing employees when the list is empty"""
    with app.app_context():
        admin = m_employee(
            first_name="Admin",
            last_name="User",
            email="admin@test.com",
            department="IT",
            joining_date=datetime.now(timezone.utc),
            is_admin=True,
            photo="default.jpg"
        )
        db.session.add(admin)
        db.session.commit()

        with client.session_transaction() as session:
            session['_user_id'] = admin.emp_id

        response = client.get('/view_employees')
        assert response.status_code == 200
        assert b"No employees found" in response.data or b"Employee List" in response.data

def test_edit_employee_phone_validation(client, test_employees, app):
    """Test phone number validation during employee edit"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        data = {
            'phone_no': '123', 
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '1990-01-01',
            'is_admin': False
        }
        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Phone number must be exactly 10 digits" in response.data
        
def test_edit_employee_successful_update(client, test_employees, app):
    """Test successful employee update with all valid fields"""
    with app.app_context():
        admin_id = test_employees[0]
        with client.session_transaction() as session:
            session['_user_id'] = admin_id

        # Get the edit form first to get CSRF token
        get_response = client.get(f'/edit_employees/{test_employees[1]}')
        assert get_response.status_code == 200

        # Extract CSRF token if present
        csrf_token = None
        if b'name="csrf_token"' in get_response.data:
            csrf_token = get_response.data.decode('utf-8').split('name="csrf_token" value="')[1].split('"')[0]

        data = {
            'phone_no': '1234567890',
            'address': '123 Test St',
            'gender': 'Male',
            'blood_group': 'O+',
            'department': 'Finance',
            'dob': '1990-01-01',
            'is_admin': False
        }
        if csrf_token:
            data['csrf_token'] = csrf_token

        response = client.post(f'/update_employees/{test_employees[1]}', data=data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee profile updated successfully" in response.data

        updated_emp = m_employee.query.get(test_employees[1])
        assert updated_emp.phone_no == '1234567890'
        assert updated_emp.address == '123 Test St'
        assert updated_emp.gender == 'Male'
        assert updated_emp.blood_group == 'O+'
        assert updated_emp.department == 'Finance'
        assert str(updated_emp.dob) == '1990-01-01'

# def test_available_routes(app):
#     with app.app_context():
#         routes = [str(rule) for rule in app.url_map.iter_rules()]
#         assert '/view_employees' in routes, f"Route /view_employees not found. Available routes: {routes}"
