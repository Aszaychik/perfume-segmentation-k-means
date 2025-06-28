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
from flask import render_template, request, redirect, current_app, url_for, session
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sqlalchemy import create_engine

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

    # Extract initial centroids in specified order
    initial_centroids = []
    for cid in centroid_ids:
        centroid_data = df_sales[df_sales['id'] == cid][variables]
        if centroid_data.empty:
            raise ValueError(f"Centroid ID {cid} not found in sales data.")
        initial_centroids.append(centroid_data.values[0])
    centroids = np.array(initial_centroids)

    # Prepare feature matrix
    X = df_sales[variables].values

    # Custom K-Means algorithm
    for _ in range(iterations):
        raw_diffs = (X[:, None, :] - centroids[None, :, :]).sum(axis=2)
        dists = np.abs(raw_diffs)
        labels = np.argmin(dists, axis=1)
        df_sales['cluster'] = labels
        
        new_centroids = np.zeros_like(centroids)
        for j in range(len(centroid_ids)):
            cluster_data = df_sales[df_sales['cluster'] == j]
            if not cluster_data.empty:
                new_centroids[j] = cluster_data[variables].mean().values
            else:
                new_centroids[j] = centroids[j]
        centroids = new_centroids

    # Store clusters in the database
    db.session.query(Result).delete()
    db.session.query(Cluster).delete()
    db.session.commit()

    # Create Cluster objects with actual values (not rounded)
    clusters = []
    for cluster_id in range(len(centroid_ids)):
        cluster_members = df_sales[df_sales['cluster'] == cluster_id]
        
        if cluster_members.empty:
            continue

        # Calculate characteristics
        age_mean = cluster_members['age'].mean()
        perfume_mode = cluster_members['perfume_id'].mode()[0]
        gender_mode = cluster_members['gender'].mode()[0]
        profession_mode = cluster_members['profession_id'].mode()[0]

        new_cluster = Cluster(
            label=f"Age {round(age_mean)}",
            age=float(age_mean),
            perfume_id=float(perfume_mode),
            gender=float(gender_mode),
            profession_id=float(profession_mode)
        )
        db.session.add(new_cluster)
        clusters.append(new_cluster)
    
    db.session.commit()

    # Save results
    for _, row in df_sales.iterrows():
        cluster_id = clusters[row['cluster']].id
        new_result = Result(cluster_id=cluster_id, sales_id=row['id'])
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
        
        # Prepare full precision results
        results = []
        for cluster in clusters:
            results.append({
                'cluster': cluster.id,
                'age': cluster.age,
                'perfume_id': cluster.perfume_id,
                'gender': cluster.gender,
                'profession_id': cluster.profession_id
            })
        
        # Prepare rounded results
        results_rounded = []
        for cluster in clusters:
            results_rounded.append({
                'cluster': cluster.id,
                'age': round(cluster.age),
                'perfume_id': round(cluster.perfume_id),
                'gender': round(cluster.gender),
                'profession_id': round(cluster.profession_id)
            })
        
        # Prepare rounded results with names
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
                'age': round(cluster.age),
                'perfume_id': round(cluster.perfume_id),
                'gender': round(cluster.gender),
                'profession_id': round(cluster.profession_id),
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
