# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.cluster import blueprint
from apps.cluster.models import Cluster, Result
from apps import db
from apps.sale.models import Sale
from apps.perfume.models import Perfume
from apps.profession.models import Profession
from flask import render_template, request, redirect, current_app, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
import numpy as np
import pandas as pd

def perform_kmeans(iterations, variables, centroid_ids=[5,10,15,20,25]):
    # Fetch sales data with perfume and profession names
    query = db.session.query(
        Sale.id,
        Sale.age,
        Sale.gender,
        Sale.profession_id,
        Sale.perfume_id,
        Perfume.name.label('perfume_name'),
        Profession.name.label('profession_name')
    ).join(Perfume, Sale.perfume_id == Perfume.id
    ).join(Profession, Sale.profession_id == Profession.id)
    
    sales_data = query.all()
    if not sales_data:
        raise ValueError("No sales data available for clustering.")

    # Convert sales data to DataFrame
    df_sales = pd.DataFrame(sales_data, columns=[
        'id', 'age', 'gender', 'profession_id', 'perfume_id',
        'perfume_name', 'profession_name'
    ])

    # Prepare feature matrix
    X = df_sales[variables].values
    k = len(centroid_ids)

    # Initialize centroids based on specified IDs (1-based labels)
    centroids = (
        df_sales.set_index('id')
                .loc[centroid_ids, variables]
                .to_numpy()
    )

    # Run iterations of custom K-Means
    for _ in range(iterations):
        # Compute distances using absolute-sum (as in notebook)
        raw_diffs = (X[:, None, :] - centroids[None, :, :]).sum(axis=2)
        dists = np.abs(raw_diffs)
        # Assign clusters (1-based labels)
        labels = np.argmin(dists, axis=1) + 1
        df_sales['cluster'] = labels

        # Update centroids by mean of assigned points, reindex to maintain order 1..k
        df_means = (
            df_sales.groupby('cluster')[variables]
                    .mean()
                    .reindex(range(1, k+1))
        )
        # Replace missing means with previous centroids
        updated = np.where(np.isnan(df_means.values), centroids, df_means.values)
        centroids = updated

    # Clear previous results and clusters
    db.session.query(Result).delete()
    db.session.query(Cluster).delete()
    db.session.commit()

    # Persist new clusters and results
    clusters = []
    for label in range(1, k+1):
        members = df_sales[df_sales['cluster'] == label]
        if members.empty:
            continue
        # Compute actual means (float)
        means = members[variables].mean()
        cluster_obj = Cluster(
            label=f"Cluster {label}",
            age=float(means['age']),
            perfume_id=float(means['perfume_id']),
            gender=float(means['gender']),
            profession_id=float(means['profession_id'])
        )
        db.session.add(cluster_obj)
        clusters.append(cluster_obj)
    db.session.commit()

    # Save results linking sales to cluster IDs
    for _, row in df_sales.iterrows():
        # Find matching cluster object by label suffix
        target = next((c for c in clusters if c.label.endswith(str(row['cluster']))), None)
        if target:
            result = Result(cluster_id=target.id, sales_id=row['id'])
            db.session.add(result)
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

        # Enforce required variables
        required_vars = {'perfume_id', 'age', 'gender', 'profession_id'}
        missing_vars = required_vars - set(variables)
        
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        if not variables:
            raise ValueError("No variables selected")
            
        if iterations < 1 or iterations > 100:
            raise ValueError("Iterations must be between 1-100")
        
        success = perform_kmeans(iterations, variables)
        
        if success:
            return redirect(url_for('cluster_blueprint.cluster_table'))
        else:
            return render_template('home/page-500.html'), 500
            
    except Exception as e:
        current_app.logger.error(f"Cluster error: {str(e)}")
        return render_template('home/page-500.html'), 500

@blueprint.route('/cluster/table')
@login_required
def cluster_table():
    try:
        # Get all clusters
        clusters = Cluster.query.all()
        
        # Prepare full precision results (actual values from database)
        results = []
        for cluster in clusters:
            results.append({
                'cluster': cluster.id,
                'age': cluster.age,
                'perfume_id': cluster.perfume_id,
                'gender': cluster.gender,
                'profession_id': cluster.profession_id
            })
        
        # Prepare rounded results (using np.floor(x + 0.5) for "round half up")
        results_rounded = []
        for cluster in clusters:
            results_rounded.append({
                'cluster': cluster.id,
                'age': int(np.floor(cluster.age + 0.5)),
                'perfume_id': int(np.floor(cluster.perfume_id + 0.5)),
                'gender': int(np.floor(cluster.gender + 0.5)),
                'profession_id': int(np.floor(cluster.profession_id + 0.5))
            })
        
        # Prepare rounded results with names (for display only)
        results_rounded_name = []
        for cluster in clusters:
            # Get most common perfume and profession names for this cluster
            perfume_name = db.session.query(
                Perfume.name
            ).join(Sale, Sale.perfume_id == Perfume.id
            ).join(Result, Result.sales_id == Sale.id
            ).filter(Result.cluster_id == cluster.id
            ).group_by(Perfume.name
            ).order_by(db.func.count().desc()
            ).first()
            
            profession_name = db.session.query(
                Profession.name
            ).join(Sale, Sale.profession_id == Profession.id
            ).join(Result, Result.sales_id == Sale.id
            ).filter(Result.cluster_id == cluster.id
            ).group_by(Profession.name
            ).order_by(db.func.count().desc()
            ).first()
            
            results_rounded_name.append({
                'cluster': cluster.id,
                'age': int(np.floor(cluster.age + 0.5)),
                'perfume_id': int(np.floor(cluster.perfume_id + 0.5)),
                'gender': int(np.floor(cluster.gender + 0.5)),
                'profession_id': int(np.floor(cluster.profession_id + 0.5)),
                'perfume_name': perfume_name[0] if perfume_name else "N/A",
                'profession_name': profession_name[0] if profession_name else "N/A"
            })
        
        return render_template('cluster/table.html', 
                              results=results,
                              results_rounded=results_rounded,
                              results_rounded_name=results_rounded_name)
            
    except Exception as e:
        current_app.logger.error(f"Cluster table error: {str(e)}")
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
