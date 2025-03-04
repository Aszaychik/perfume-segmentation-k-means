# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.sale import blueprint
from apps.sale.models import Sale
from apps.perfume.models import Perfume
from apps.profession.models import Profession
from flask import render_template, request, jsonify
from apps import db
from flask_login import login_required
from datetime import datetime
from jinja2 import TemplateNotFound


@blueprint.route('/sales')
@login_required
def index():
    sales = Sale.query.order_by(Sale.createdAt.desc()).all()


    return render_template('sale/index.html', segment='index', sales=sales)

@blueprint.route('/sales/create', methods=['POST'])
@login_required
def create_sale():
    try:
        data = request.form
        sale = Sale(
            perfume_id=data['perfume_id'],
            profession_id=data['profession_id'],
            age=data['age'],
            gender=data['gender']
        )
        db.session.add(sale)
        db.session.commit()
        return jsonify(success=True)
    except Exception as e:
        return jsonify(error=str(e)), 400

@blueprint.route('/sales/options')
@login_required
def get_options():
    perfumes = [{'id': p.id, 'name': p.name} for p in Perfume.query.all()]
    professions = [{'id': p.id, 'name': p.name} for p in Profession.query.all()]
    
    perfume_options = ''.join([f'<option value="{p["id"]}">{p["name"]}</option>' for p in perfumes])
    profession_options = ''.join([f'<option value="{p["id"]}">{p["name"]}</option>' for p in professions])
    
    return jsonify({
        'perfumes': perfume_options,
        'professions': profession_options
    })


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
