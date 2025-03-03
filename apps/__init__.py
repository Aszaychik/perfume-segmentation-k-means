# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from importlib import import_module

db = SQLAlchemy()
login_manager = LoginManager()

def register_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)

def register_blueprints(app):
    for module_name in ('authentication', 'home', 'sale'):
        module = import_module('apps.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)

def configure_database(app):
    # Import all models first
    from apps.authentication.models import Users
    from apps.perfume.models import Perfume
    from apps.profession.models import Profession
    from apps.sale.models import Sale

    @app.before_first_request
    def initialize_database():
        """Initialize database on first request"""
        with app.app_context():
            try:
                db.create_all()
                print("> Database tables created successfully!")
                
                # Auto-seed data if tables are empty
                if not Perfume.query.first():
                    from apps.perfume.commands import seed_perfumes
                    seed_perfumes()
                    
                if not Profession.query.first():
                    from apps.profession.commands import seed_professions
                    seed_professions()
                    
                if not Sale.query.first():
                    from apps.sale.commands import seed_sales
                    seed_sales()
                    
            except Exception as e:
                print(f'> Database initialization error: {str(e)}')
                
                # Fallback to SQLite
                basedir = os.path.abspath(os.path.dirname(__file__))
                app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'db.sqlite3')
                print('> Fallback to SQLite ')
                db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Initialize extensions first
    register_extensions(app)
    
    # Push app context for database operations
    with app.app_context():
        try:
            db.create_all()
        except:
            pass
    
    # Register blueprints and configure database
    register_blueprints(app)
    configure_database(app)
    
    return app