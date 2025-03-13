# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.cluster import blueprint
from apps.cluster.models import Cluster, Result
from apps import db
from apps.sale.models import Sale
from flask import render_template, request, redirect, current_app
from flask_login import login_required
from jinja2 import TemplateNotFound
import numpy as np
from sklearn.cluster import KMeans

def perform_kmeans(iterations, variables, centroid_ids=[5,10,15,20,25]):
    # Fetch sales data based on selected variables
    sales_data = Sale.query.with_entities(Sale.id, *[getattr(Sale, var) for var in variables]).all()
    
    if not sales_data:
        raise ValueError("No sales data available for clustering.")
    
    # Convert sales data to a DataFrame
    import pandas as pd
    df_sales = pd.DataFrame(sales_data, columns=['id'] + variables)

    # Extract initial centroids from the specified sales IDs
    initial_centroids = df_sales[df_sales['id'].isin(centroid_ids)][variables].values

    if len(initial_centroids) != len(centroid_ids):
        raise ValueError("Some centroid IDs were not found in sales data.")

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=len(centroid_ids), init=initial_centroids, n_init=1, max_iter=iterations)
    df_sales['cluster'] = kmeans.fit_predict(df_sales[variables])

    # Store clusters in the database
    db.session.query(Cluster).delete()  # Clear previous cluster data
    db.session.query(Result).delete()   # Clear previous result data
    db.session.commit()

    # Save cluster data
    clusters = []
    for cluster_id in range(len(centroid_ids)):
        # Get cluster age (mean of the age in that cluster)
        cluster_label = round(df_sales[df_sales['cluster'] == cluster_id]['age'].mean())
        new_cluster = Cluster(label=cluster_label)
        db.session.add(new_cluster)
        clusters.append(new_cluster)

    db.session.commit()

    # Save result data
    for index, row in df_sales.iterrows():
        sales_id = row['id']
        cluster_id = row['cluster']
        new_result = Result(cluster_id=clusters[cluster_id].id, sales_id=sales_id)
        db.session.add(new_result)

    db.session.commit()
    
    return True

@blueprint.route('/cluster')
@login_required
def index():

    return render_template('cluster/index.html')


@blueprint.route('/cluster/process', methods=['POST'])
@login_required
def process_cluster():
    try:
        iterations = int(request.form.get('cluster_count'))
        variables = request.form.getlist('variables')
        
        if not variables:
            raise ValueError("No variables selected")
            
        if iterations < 1 or iterations > 100:
            raise ValueError("Iterations must be between 1-100")
        
        success = perform_kmeans(iterations, variables)
        
        if success:
            return redirect('/cluster/results')
        else:
            return render_template('home/page-500.html'), 500
            
    except Exception as e:
        current_app.logger.error(f"Cluster error: {str(e)}")
        return render_template('home/page-500.html'), 500


@blueprint.route('/cluster/results')
@login_required
def cluster_results():
    clusters = Cluster.query.all()
    results = []
    for cluster in clusters:
        sales_count = Result.query.filter_by(cluster_id=cluster.id).count()
        results.append({
            'id': cluster.id,
            'label': cluster.label,
            'count': sales_count
        })
    return render_template('cluster/results.html', results=results)

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
