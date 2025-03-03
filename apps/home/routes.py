# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, redirect, url_for
from jinja2 import TemplateNotFound
from flask_login import login_required, current_user
from apps import db

@blueprint.route('/')
@blueprint.route('/index')
def index():
    return render_template('pages/index.html', segment='dashboard', parent="dashboard")

@blueprint.route('/billing')
def billing():
    return render_template('pages/billing.html', segment='billing')

@blueprint.route('/rtl')
def rtl():
    return render_template('pages/rtl.html', segment='rtl')

@blueprint.route('/tables')
def tables():
    return render_template('pages/tables.html', segment='tables')

@blueprint.route('/virtual_reality')
def virtual_reality():
    return render_template('pages/virtual-reality.html', segment='virtual_reality')


@blueprint.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        address = request.form.get('address')
        bio = request.form.get('bio')

        current_user.first_name = first_name
        current_user.last_name = last_name
        current_user.address = address
        current_user.bio = bio

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()

        return redirect(url_for('home_blueprint.profile'))

    return render_template('pages/profile.html', segment='profile')


# Helper - Extract current page name from request
@blueprint.app_template_filter('replace_value')
def replace_value(value, args):
  return value.replace(args, " ").title()

def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
