# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.cluster import blueprint
from flask import render_template, request, redirect
from flask_login import login_required
from jinja2 import TemplateNotFound


@blueprint.route('/cluster')
@login_required
def index():

    return render_template('cluster/index.html')

@blueprint.route('/cluster/process', methods=['POST'])
@login_required
def process_cluster():
    try:
        cluster_count = int(request.form.get('cluster_count'))
        variables = request.form.getlist('variables')
        
        # Add your clustering logic here
        # Example: perform_kmeans(cluster_count, variables)
        
        return redirect('/cluster/results')
    except Exception as e:
        return render_template('home/page-500.html'), 500

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
