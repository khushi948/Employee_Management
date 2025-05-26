import pytest
from app import create_app, db
from app.models.models import m_employee

@pytest.fixture
def app():
    try:
        app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': 'postgresql://postgres:admin@localhost/test_employee_db',
            'WTF_CSRF_ENABLED': False
        })

        with app.app_context():
            try:
                db.drop_all()
                db.create_all()
                inspector = db.inspect(db.engine)
                tables = inspector.get_table_names()
                print("Database tables created successfully")
            except Exception as e:
                print(f"Error creating database tables: {str(e)}")
                raise e

            yield app
    except Exception as e:
        print(f"Error in app fixture: {str(e)}")
        raise e

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()
