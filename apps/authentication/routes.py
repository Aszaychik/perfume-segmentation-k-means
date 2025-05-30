# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, redirect, request, url_for, jsonify
from flask_login import (
    current_user,
    login_user,
    logout_user,
    login_required
)

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users

from apps.authentication.util import verify_pass


@blueprint.route('/')
def route_default():
    return redirect(url_for('authentication_blueprint.login'))


# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = Users.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):

            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))

        # Something (user or pass) is not ok
        return render_template('accounts/login.html',
                               msg='Wrong user or password',
                               form=login_form)

    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username already registered',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email already registered',
                                   success=False,
                                   form=create_account_form)

        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        # Delete user from session
        logout_user()
        
        return render_template('accounts/register.html',
                               msg='User created successfully.',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)

@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))

# User Management
@blueprint.route('/accounts')
@login_required
def accounts():
    # Check if user is admin
    if current_user.role != 'admin':
        return render_template('home/page-403.html'), 403
    
    # Get all users from database
    users = Users.query.all()
    return render_template('accounts/index.html', users=users)

@blueprint.route('/accounts/create' , methods=['POST'])
@login_required
def create():
    try:
        # Check if user is admin
        if current_user.role != 'admin':
            return render_template('home/page-403.html'), 403

        # Create a new user
        user = Users(
            username=request.form['username'],
            email=request.form['email'],
            password=request.form['password'],
            role=request.form['role']
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('authentication_blueprint.accounts'))
    except Exception as e:
        return render_template('home/page-500.html', error=str(e)), 500


@blueprint.route('/accounts/delete/<int:user_id>', methods=['POST'])
@login_required
def delete(user_id):
    try:
        # Check if user is admin
        if current_user.role != 'admin':
            return render_template('home/page-403.html'), 403
        # Delete user
        user = Users.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 400

@blueprint.route('/accounts/update/<int:user_id>', methods=['POST'])
@login_required
def update(user_id):
    try:
        # Check if user is admin
        if current_user.role != 'admin':
            return render_template('home/page-403.html'), 403

        user = Users.query.get_or_404(user_id)
        
        # Check username uniqueness
        if Users.query.filter(Users.username == request.form['username'], Users.id != user_id).first():
            return jsonify(error='Username already exists'), 400
            
        # Check email uniqueness
        if Users.query.filter(Users.email == request.form['email'], Users.id != user_id).first():
            return jsonify(error='Email already exists'), 400

        # Update user
        user.username = request.form['username']
        user.email = request.form['email']
        user.role = request.form['role']
        
        # Update password if provided
        if request.form.get('password'):
            user.password = request.form['password']
            
        db.session.commit()
        return redirect(url_for('authentication_blueprint.accounts'))
    except Exception as e:
        return render_template('home/page-500.html', error=str(e)), 500

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
